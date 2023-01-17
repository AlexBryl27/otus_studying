from fastapi import FastAPI, Query
from typing import List

from engine import ClassificationAPI
from definitions import DATA_DIR
import contract


app = FastAPI(title='Classification API')
engine = ClassificationAPI(DATA_DIR / 'model.pkl')


@app.get('/predict', response_model=contract.Response)
def predict(request: List[str] = Query(default=None)):

    return engine.process(request)
