from langchain.tools import tool

@tool
def buffer_analysis(table:str,distance:int):

    sql = f"""
    SELECT ST_Buffer(geom,{distance})
    FROM {table}
    """

    return sql
