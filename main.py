from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib  # Для сохранения и загрузки модели
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 1. Загрузка обученной модели (если она есть):

try:
    model = joblib.load("precipitation_model.joblib")
    print("Модель загружена из файла precipitation_model.joblib")
except FileNotFoundError:
    print("Файл precipitation_model.joblib не найден. Обучается новая модель.")

    # Создание синтетических данных для примера (если нет сохраненной модели)
    import numpy as np

    np.random.seed(42)
    n_samples = 100
    temperature = np.random.uniform(5, 30, n_samples)
    humidity = np.random.uniform(30, 90, n_samples)
    wind_speed = np.random.uniform(0, 15, n_samples)
    cloud_cover = np.random.uniform(0, 100, n_samples)
    precipitation = (0.05 * humidity + 0.01 * cloud_cover - 0.02 * wind_speed + np.random.normal(0, 0.5, n_samples)).clip(min=0)

    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'precipitation': precipitation
    })

    # Разделение данных на признаки и целевую переменную
    X = data[['temperature', 'humidity', 'wind_speed', 'cloud_cover']]
    y = data['precipitation']

    # Обучение модели
    model = LinearRegression()
    model.fit(X, y)

    # Сохранение обученной модели
    joblib.dump(model, "precipitation_model.joblib")
    logger.info("Модель обучена и сохранена в файл precipitation_model.joblib")

# 2.  Определение входных данных (используем Pydantic BaseModel):
class WeatherData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
# 3.  Определение endpoint для прогнозирования:
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "OK"}

@app.post("/predict")
async def predict_precipitation(weather_data: WeatherData):
    """
    Предсказывает количество осадков на основе заданных параметров погоды.
    Args:
        weather_data: Объект WeatherData, содержащий температуру, влажность, скорость ветра и облачность.
    Returns:
        Словарь с предсказанным количеством осадков.
    """
    try:
        # Преобразование данных в DataFrame (требуется для sklearn)
        input_data = pd.DataFrame([weather_data.model_dump()])  # Важно: оборачиваем model_dump в список
        logger.info(f"Входные данные: {input_data}")  # Log DEBUG
        # Прогнозирование с помощью загруженной модели
        prediction = model.predict(input_data)[0]
        logger.info(f"Предсказанное значение осадков: {prediction}")  # Log INFO
        # Возврат результата
        return {"precipitation": prediction}
    except Exception as e:
        logger.exception("Произошла ошибка при прогнозировании.")  # Log ERROR с трассировкой стека
        raise HTTPException(status_code=500, detail=str(e))  # Обработка ошибок

