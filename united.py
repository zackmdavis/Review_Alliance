import logging
import time

import pycld2
import IPython
import matplotlib.pyplot as plot
import spacy
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sqlalchemy import (Column, ForeignKey, Integer, MetaData, String, Table,
                        Text, create_engine)
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from spacy_cld import LanguageDetector

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s • %(levelname)s—%(message)s")

engine = create_engine("mysql://localhost/yelp_db")

metadata = MetaData()

metadata.reflect(engine, only=['business'])

# the Yelp SQL-dump is pretty dumb and doesn't have any foreign key constraints
# for SQLAlchemy to discover, so we manually describe the 'category' and
# 'review' tables so that we can get the Review ↔ Business ↔ Category
# relationships.

Table('category', metadata,
      Column('id', Integer, primary_key=True),
      Column('category', String(255)),
      Column('business_id', String(22), ForeignKey('business.id')))

Table('review', metadata,
      Column('id', Integer, primary_key=True),
      Column('stars', Integer),
      Column('useful', Integer),
      Column('text', Text),
      Column('business_id', String(22), ForeignKey('business.id')))

Base = automap_base(metadata=metadata)
Base.prepare()

Category = Base.classes.category
Review = Base.classes.review
Business = Base.classes.business

session = Session(engine)

nlp = spacy.load('en_core_web_lg')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)

class ChauvinismException(Exception):
    pass

class ReviewVec:
    def __init__(self, vector, text, stars, useful, categories):
        self.vector = vector
        self.text = text
        self.stars = stars
        self.useful = useful
        self.categories = categories

    @classmethod
    def from_review(cls, review):
        doc = nlp(review.text)
        if doc._.languages != ["en"]:
            raise ChauvinismException("detected language {}, expected English")
        vector = doc.vector
        categories = [c.category for c in review.business.category_collection]
        return cls(vector, review.text, review.stars, review.useful, categories)


def grab_reviewvecs(how_many):
    last_tick = time.time()
    reviewvecs = []
    for i, review in enumerate(session.query(Review).limit(how_many)):
        try:
            reviewvec = ReviewVec.from_review(review)
        except ChauvinismException as ce:
            # There were 9 French, Spanish, or German reviews in the first
            # thousand I looked at, which really threw off the first reduced
            # word vector dimension
            logging.warning("filtering out non-English review: %r...", review.text[:50])
            continue
        except pycld2.error:
            # workaround for nickdavidhaynes/spacy-cld#1, upstream fix not yet
            # on PyPI
            logging.exception("hit a bug while processing review text %r",
                              review.text)
            continue
        reviewvecs.append(reviewvec)
        tick = time.time()
        if tick - last_tick > 2:
            last_tick = tick
            # looks like we're doing about 5 reviews/sec
            logging.info("converted %s reviews of %s so far", i, how_many)
    return reviewvecs


def reduce_reviewvec_dimensionality(reviewvecs, n=10):
    pca = PCA(n_components=n)
    vectors = [r.vector for r in reviewvecs]
    pca.fit(vectors)
    print("explained variance fractions", pca.explained_variance_ratio_)
    for i, reduced in enumerate(pca.transform(vectors)):
        reviewvecs[i].vector = reduced


def plot_by_stars(reviewvecs):
    figure = plot.figure(figsize=(4, 3))
    axes = Axes3D(figure)

    xs = []
    ys = []
    zs = []
    stars = []
    for rv in reviewvecs:
        xs.append(rv.vector[0])
        ys.append(rv.vector[1])
        zs.append(rv.vector[2])
        stars.append(rv.stars)

    axes.scatter(
        xs, ys, zs, c=stars,
        cmap=ListedColormap(['#302080', '#604060', '#906040',
                             '#c08020', '#f0a000'])
    )
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")

    plot.show()


if __name__ == "__main__":
    rvs = grab_reviewvecs(2500)
    reduce_reviewvec_dimensionality(rvs)
    rvs_by_1st = sorted(rvs, key=lambda rv: rv.vector[0])
    rvs_by_2nd = sorted(rvs, key=lambda rv: rv.vector[1])
    rvs_by_3rd = sorted(rvs, key=lambda rv: rv.vector[2])

    # drop into an IPython shell for exploration
    IPython.embed()
