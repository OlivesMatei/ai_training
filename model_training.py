from tensorflow.keras.callbacks import ModelCheckpoint
import data_preprocessing as dp
import model_building as mb
def save_model(model, scaler, model_save_path):
    model.save(model_save_path)

def train_model(data_filepath, model_save_path):
    data = dp.load_data(data_filepath)
    data = dp.clean_data(data)
    X_train, X_test, y_train, y_test = dp.split_data(data, ['diabetes', 'heart_disease', 'alzheimer'])
    X_train_scaled, X_test_scaled, scaler = dp.scale_data(X_train, X_test)
    
    model = mb.build_model(X_train_scaled.shape[1])
    
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min')
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, callbacks=[checkpoint])
    
    return model, scaler
