import requests, pandas as pd, numpy as np

class BlockchainDataBridge:
    def __init__(self, source='ethereum', api_url=None, file_path=None):
        self.source = source
        self.api_url = api_url or 'https://api.blockchair.com/ethereum/blocks'
        self.file_path = file_path
        self.real_data = None
        try:
            if self.source == 'ethereum':
                df = self.fetch_ethereum_data()
                if df is not None:
                    self.real_data = self.extract_performance_metrics(df)
        except Exception as e:
            print('Data bridge error:', e)

    def fetch_ethereum_data(self, blocks=30):
        resp = requests.get(self.api_url, params={'limit': blocks})
        data = resp.json()
        recs = []
        for b in data.get('data', []):
            recs.append({
                'id': b.get('id'),
                'tx_count': b.get('transaction_count', 0),
                'gas_used': b.get('gas_used', 0),
                'time': b.get('time', 1),
                'size': b.get('size', 0)
            })
        if not recs:
            return None
        return pd.DataFrame(recs)

    def load_from_csv(self, path):
        df = pd.read_csv(path)
        return self.extract_performance_metrics(df)

    def extract_performance_metrics(self, df):
        df = df.copy()
        df['time_diff'] = df['time'].diff().fillna(1.0)
        df['throughput'] = df['tx_count'] / (df['time_diff'] + 1e-8)
        df['efficiency'] = 1.0 - (df['gas_used'] / (df['gas_used'].max() + 1e-8))
        df['scalability'] = df['throughput'] / (df['throughput'].max() + 1e-8)
        df['security'] = np.exp(-df['size'] / (df['size'].max() + 1e-8))
        df['decentralization'] = np.random.uniform(0.6, 0.9, len(df))
        return df[['scalability','decentralization','security','efficiency']]
