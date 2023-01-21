import ccxt
from ccxt.base.errors import RequestTimeout
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time

exchanges = {
    "binance": ccxt.binance(),
    "yobit": ccxt.yobit(),
    "bittrex": ccxt.bittrex(),
    "kucoin": ccxt.kucoin()
}

def to_timestamp(exchange, dt):
    return exchanges[exchange].parse8601(dt.isoformat())

def download(exchange, symbol, start, end):
    '''
    Download all the transaction for a given symbol from the start date to the end date
    @param symbol: the symbol of the coin for which download the transactions
    @param start: the start date from which download the transaction
    @param end: the end date from which download the transaction
    '''

    records = []
    since = start
    ten_minutes = 60000 * 10

    print('Downloading {} from {} to {}'.format(symbol, exchanges[exchange].iso8601(start), exchanges[exchange].iso8601(end)))

    while since < end:
        # print('since: ' + binance.iso8601(since)) #uncomment this line of code for verbose download
        try:
            orders = exchanges[exchange].fetch_trades(symbol + '-BTC', since)
        except RequestTimeout:
            time.sleep(5)
            orders = exchanges[exchange].fetch_trades(symbol + '-BTC', since)

        if len(orders) > 0:

            latest_ts = orders[-1]['timestamp']
            if since != latest_ts:
                since = latest_ts
            else:
                since += ten_minutes

            for l in orders:
                records.append({
                    'symbol': l['symbol'],
                    'timestamp': l['timestamp'],
                    'datetime': l['datetime'],
                    'side': l['side'],
                    'price': l['price'],
                    'amount': l['amount'],
                    'base_volume': float(l['price']) * float(l['amount']),
                })
        else:
            since += ten_minutes

    return pd.DataFrame.from_records(records)


def download_from_exchange(exchange, base="BTC", days_before=7, days_after=7):
    '''
    Download all the transactions for all the pumps in binance in a given interval
    @param days_before: the number of days before the pump
    @param days_after: the number of days after the pump
    '''

    df = pd.read_csv('./data/pump_telegram.csv')
    filtered = df[df['exchange'] == exchange]

    for i, pump in filtered.iterrows():
        symbol = pump['symbol']
        date = pump['date'] + ' ' + pump['hour']
        pump_time = datetime.strptime(date, "%Y-%m-%d %H:%M")
        before = to_timestamp(exchange, pump_time - timedelta(days=days_before))
        after = to_timestamp(exchange, pump_time + timedelta(days=days_after))
        # to comment out
        import os
        if os.path.exists('data/full/{}_{}_{}_{}'.format(exchange, symbol, base, str(date).replace(':', '.') + '.csv')):
            print(symbol)
            continue
        #
        df = download(exchange, symbol, before, after)
        df.to_csv('data/full/{}_{}_{}_{}'.format(exchange, symbol, base, str(date).replace(':', '.') + '.csv'), index=False)


if __name__ == '__main__':
    days_before = 14
    days_after = 14
    # download_from_exchange("binance", days_before=days_before, days_after=days_after)
    # download_from_exchange("yobit", days_before=days_before, days_after=days_after)
    download_from_exchange("bittrex", days_before=days_before, days_after=days_after)
