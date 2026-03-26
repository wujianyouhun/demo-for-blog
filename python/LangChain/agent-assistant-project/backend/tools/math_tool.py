
def calc(q):
    try:
        expr = q.replace("计算","")
        return str(eval(expr))
    except:
        return "计算失败"
