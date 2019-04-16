from kerasImplTimeSeries import network

def create(feature_size, input_length, output_length, N_UNITS, DROPOUT):
    """
    creates all models as they exist in network.py
    :return: The compiled model and the modelname
    """
    model1, name1 = network.define_model_15(feature_size, input_length, output_length, N_UNITS, DROPOUT)
    model1.compile(optimizer='adam', loss='mean_squared_error')  # TODO appropriate loss function

    #model2, name2 = network.define_model_2(feature_size, input_length, output_length, N_UNITS, DROPOUT)
    #model2.compile(optimizer='adam', loss='mean_squared_error')

    #model3, name3 = network.define_model_3(feature_size, input_length, output_length, N_UNITS, DROPOUT)
    #model3.compile(optimizer='adam', loss='mean_squared_error')

    return [(model1, name1)]