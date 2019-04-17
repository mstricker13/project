from kerasImplTimeSeries import network

def create(feature_size, input_length, output_length):
    """
    creates all models as they exist in network.py
    :return: The compiled model and the modelname
    """
    models = list()
    for horizon in output_length:
        model1, name1 = network.define_model_1(feature_size, input_length, horizon)
        model1.compile(optimizer='adam', loss='mean_squared_error')  # TODO appropriate loss function
        models.append((model1, name1, horizon))

        #model2, name2 = network.define_model_2(feature_size, input_length, output_length, N_UNITS, DROPOUT)
        #model2.compile(optimizer='adam', loss='mean_squared_error')

        #model3, name3 = network.define_model_3(feature_size, input_length, output_length, N_UNITS, DROPOUT)
        #model3.compile(optimizer='adam', loss='mean_squared_error')

    return models