
async def ingest_file(file, user_id):
    content = await file.read()
    # TODO: parse + embedding + store
    return True
