import IPython
from sqlalchemy import (Column, ForeignKey, Integer, MetaData, String, Table,
                        create_engine)
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session


engine = create_engine("mysql://localhost/yelp_db")

metadata = MetaData()

metadata.reflect(engine, only=['business', 'review'])

Table('category', metadata,
      Column('id', Integer, primary_key=True),
      Column('category', String(255)),
      Column('business_id', String(22), ForeignKey('business.id')))

Base = automap_base(metadata=metadata)
Base.prepare()

Category = Base.classes.category
Review = Base.classes.review
Business = Base.classes.business

session = Session(engine)


if __name__ == "__main__":
    # drop into an IPython shell for exploration
    IPython.embed()
