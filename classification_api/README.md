# Classification API

Учебный проект в рамках курса OTUS Advanced ML

## Цель:
В этом домашнем задании, вам нужно создать fastapi-приложение для вашей модели классификации. И развернуть данное приложение локально при помощи Docker. Протестировать get запросы (направляя X-вектор переменных) и получить response в виде целевой переменной (для теста можно использовать Postman).

## Датасет:
Датасет доступен по ссылке: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

## Сборка и запуск:
```
docker build --no-cache --pull -t clfapi .
docker run -p 80:8000 -d -t clfapi
```

## Пример запроса:
```python
import requests

x = [67., 0., 2., 152., 277., 0., 0., 172., 0., 0., 0., 1., 0.]
data = {"request": x}
response = requests.get('http://127.0.0.1:80/predict', params=data)
```