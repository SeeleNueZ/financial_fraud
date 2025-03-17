from dataset.data_preprocess import ccfd
from method import kmeans

if __name__ == "__main__":
    c1 = [x for x in range(0, 6)]
    c2 = [x for x in range(6, 10)]
    c3 = [x for x in range(10, 20)]
    c4 = [x for x in range(20, 28)]
    config = {0: c1, 1: c2, 2: c3, 3: c4}
    train_data, test_data, train_data_val, test_data_val, data_agg, data_agg_val = ccfd(config=config)
    num_data = train_data[0].shape[0]
    num_dim = 28
    num_client = 4

    Base = kmeans.kmeans(train_data=train_data, train_data_val=train_data_val, test_data=test_data,
                         test_data_val=test_data_val, data_agg=data_agg, data_agg_val=data_agg_val, n_cluster=2,
                         config=config, num_dim=num_dim, num_data=num_data, num_client=4)

    Base.init_client()
    print(num_data)
    Base.run()
    Base.cal()
