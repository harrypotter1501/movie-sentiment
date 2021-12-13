# train models



from utils.models import load_data, vectorize, lr

from config import *


def train(df, data_path=pp_path, model_path=m_path):
    df = load_data(df=df, path=data_path)
    df = vectorize(df)
    return lr(df)

