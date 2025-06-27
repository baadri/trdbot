# AI Trading Signal Bot

Интеллектуальная система анализа криптовалютного рынка на основе ансамбля нейросетевых моделей для генерации высокоточных торговых сигналов.

## 🚀 Особенности

- **Ансамбль из 5 специализированных моделей**: LSTM, CNN, Transformer, XGBoost, BERT
- **Мультитаймфреймовый анализ**: 5m, 15m, 1h, 4h, 1d
- **Топ-20 криптовалютных пар** по рыночной капитализации
- **Продвинутый риск-менеджмент** с множественными уровнями Take Profit
- **Telegram уведомления** с детальной аналитикой
- **Веб-дашборд** в режиме реального времени
- **Комплексный бэктестинг** и оптимизация параметров

## 📊 Архитектура ансамбля

1. **LSTM_PricePredictor** - Прогнозирование движения цены
2. **CNN_PatternRecognizer** - Распознавание графических паттернов
3. **Transformer_TrendAnalyzer** - Анализ трендов и зависимостей
4. **XGBoost_VolumeAnalyzer** - Анализ объемов торгов
5. **BERT_SentimentAnalyzer** - Анализ настроений рынка

## 🛠 Установка

### Требования
- Python 3.9+
- TA-Lib библиотека
- API ключи Bybit
- Telegram Bot Token

### Пошаговая установка

1. **Клонирование репозитория**
```bash
git clone https://github.com/baadri/trdbot.git
cd trdbot
```

2. **Установка зависимостей**
```bash
# Установка TA-Lib (для Ubuntu/Debian)
sudo apt-get install libta-lib-dev

# Установка Python зависимостей
pip install -r requirements.txt
```

3. **Конфигурация**
```bash
# Скопировать файлы конфигурации
cp config/settings.yaml.example config/settings.yaml
cp config/trading_params.yaml.example config/trading_params.yaml

# Отредактировать настройки
nano config/settings.yaml
```

4. **Настройка API ключей**
```yaml
# В файле config/settings.yaml
api:
  bybit:
    testnet: true  # Установить false для продакшена
    api_key: "YOUR_BYBIT_API_KEY"
    api_secret: "YOUR_BYBIT_API_SECRET"
  
  telegram:
    bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id: "YOUR_TELEGRAM_CHAT_ID"
```

## 🎯 Запуск

### Первый запуск - Обучение моделей
```bash
# Обучение всех моделей (может занять несколько часов)
python train_models.py
```

### Запуск торгового бота
```bash
# Основной режим
python main.py

# Или в фоновом режиме
nohup python main.py > logs/bot.log 2>&1 &
```

### Запуск веб-дашборда
```bash
# Дашборд запускается автоматически на порту 8050
# Откройте в браузере: http://localhost:8050
```

## 📈 Формат торговых сигналов

```
🚨 НОВЫЙ СИГНАЛ 🚨

📊 Пара: BTC/USDT
📈 Действие: ПОКУПКА
💰 Вход: $45,230
🛑 Стоп-лосс: $44,780 (-1%)
🎯 Take Profit 1: $46,130 (+2%)
🎯 Take Profit 2: $47,030 (+4%)
🎯 Take Profit 3: $48,380 (+7%)

📊 Уверенность: 87%
⏰ Таймфрейм: 1H
💼 Размер позиции: 0.022 BTC

📝 Обоснование:
✅ Пробой уровня сопротивления
✅ Увеличение объемов (+45%)
✅ RSI развернулся от перепроданности
✅ MACD пересечение
✅ Позитивные настроения рынка
```

## 🔧 Конфигурация

### Основные параметры торговли
```yaml
# config/trading_params.yaml
risk_management:
  max_risk_per_trade: 0.02  # 2% от депозита
  max_daily_risk: 0.06      # 6% от депозита
  stop_loss:
    percentage: 0.01  # 1% стоп-лосс
  take_profit:
    tp1: 0.015  # 1.5% (R:R 1:1.5)
    tp2: 0.03   # 3% (R:R 1:3)
    tp3: 0.05   # 5% (R:R 1:5)
```

### Настройки моделей
```yaml
# config/settings.yaml
models:
  confidence_threshold: 75.0  # Минимальная уверенность
  ensemble_weights:
    lstm: 0.25
    cnn: 0.20
    transformer: 0.25
    xgboost: 0.15
    sentiment: 0.15
```

## 📊 Веб-дашборд

Дашборд включает:
- 📈 Графики в реальном времени с отметками сигналов
- 📋 История сигналов с результатами
- 💼 Текущие открытые позиции
- 📊 Статистика эффективности
- 🔥 Тепловая карта корреляций
- 📈 Метрики производительности

## 🧪 Бэктестинг

```bash
# Запуск бэктестинга
python backtesting/backtest_engine.py --start-date 2023-01-01 --end-date 2024-01-01

# Оптимизация параметров
python backtesting/optimization.py --optimize-risk-management
```

## 📈 Ключевые метрики

- **Sharpe Ratio**: > 1.5
- **Win Rate**: > 60%
- **Profit Factor**: > 2.0
- **Maximum Drawdown**: < 20%
- **Risk:Reward**: 1:1.5 - 1:5

## 🏗 Структура проекта

```
crypto_ai_bot/
├── config/                    # Конфигурационные файлы
├── data_pipeline/             # Сбор и обработка данных
├── feature_engineering/       # Инженерия признаков
├── ml_models/                 # Машинное обучение
├── trading_logic/             # Торговая логика
├── visualization/             # Визуализация и дашборд
├── notifications/             # Уведомления
├── backtesting/              # Бэктестинг
├── logs/                     # Логи
├── data/                     # Данные и кэш
├── models/                   # Обученные модели
├── main.py                   # Главный файл
└── train_models.py          # Обучение моделей
```

## 🛡 Безопасность

- ✅ API ключи в защищенных конфигурационных файлах
- ✅ Валидация всех входящих данных
- ✅ Ограничения на количество сигналов
- ✅ Мониторинг аномальной активности
- ✅ Автоматическое отключение при критических ошибках

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Создайте Pull Request

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. [LICENSE](LICENSE) файл.

## ⚠️ Disclaimer

Этот бот предназначен только для образовательных целей. Торговля криптовалютами связана с высоким риском потери средств. Всегда тестируйте на демо-счете перед использованием реальных средств.

## 📞 Поддержка

- 📧 Email: support@example.com
- 💬 Telegram: @your_support_channel
- 🐛 Issues: [GitHub Issues](https://github.com/baadri/trdbot/issues)

---

**Сделано с ❤️ для криптотрейдинг сообщества**
