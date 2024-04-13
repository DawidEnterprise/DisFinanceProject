from locust import HttpUser, task, between
from datetime import datetime
import random

class StockPriceUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def analyze_stock_price(self):
        # generate random stock symbol
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        stock_symbol = random.choice(symbols)

        # generate random start and end dates
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')

        response = self.client.post("/", data={
            "stock_symbol": stock_symbol,
            "start_date": start_date,
            "end_date": end_date
        })
        print(response.status_code)