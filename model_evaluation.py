from sklearn.metrics import classification_report
import data_preprocessing as dp
import tensorflow as tf

def evaluate_model(model_path, data_filepath):
    model = tf.keras.models.load_model(model_path)
    data = dp.load_data(data_filepath)
    data = dp.clean_data(data)
    X_train, X_test, y_train, y_test = dp.split_data(data, ['diabetes', 'heart_disease', 'alzheimer'])
    X_train_scaled, X_test_scaled, _ = dp.scale_data(X_train, X_test)
    
    y_pred = model.predict(X_test_scaled)
    y_pred = (y_pred > 0.5).astype(int)
    
    report = classification_report(y_test, y_pred, target_names=['diabetes', 'heart_disease', 'alzheimer'])
    print(report)
