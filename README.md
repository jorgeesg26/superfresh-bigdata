# Sistema de Predicción de Ventas - SuperFresh

## Descripción

Este proyecto consiste en el desarrollo de un sistema de predicción de ventas para la cadena de supermercados SuperFresh, utilizando técnicas de Big Data y Machine Learning.

El objetivo es mejorar la gestión del inventario mediante la predicción de la demanda futura de productos.

---

## Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- AWS S3
- Matplotlib

---

## Arquitectura del sistema

El sistema sigue una arquitectura sencilla de Big Data:

- Los datos se almacenan en AWS S3
- Se procesan utilizando Python
- Se entrena un modelo de Machine Learning (Random Forest)
- Se generan predicciones de ventas

---

## Dataset

Se ha utilizado un dataset simulado que incluye:

- Fecha
- Tienda
- Producto
- Ventas
- Promociones
- Temperatura

---

## Modelo

Se ha implementado un modelo de Random Forest para predecir la demanda de productos.

---

## Evaluación

El modelo ha sido evaluado utilizando las siguientes métricas:

- MAE
- RMSE
- R²

Los resultados obtenidos muestran un buen rendimiento del modelo.

---

## Resultados

El sistema permite:

- Analizar ventas históricas
- Visualizar patrones de consumo
- Generar predicciones de demanda

---

## Ejecución

Para ejecutar el proyecto:

```bash
pip install -r requirements.txt
python main.py
```
---
## Autor
Proyecto realizado por Jorge Salguero Abad
