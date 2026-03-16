from langchain.tools import tool
from db.connection import run_sql

@tool
def run_postgis_query(sql:str):
    '''Execute PostGIS SQL'''
    return run_sql(sql)
