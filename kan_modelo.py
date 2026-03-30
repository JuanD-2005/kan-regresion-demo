import torch
from kan import KAN, create_dataset
import time


# 1. PREPARACIÓN DE DATOS

def preparar_datos():
    """
    Crea un conjunto de datos sintético para un problema de regresión.
    La red intentará aprender la función subyacente: f(x, y) = exp(sin(pi*x) + y^2)
    """
    # Función matemática objetivo a modelar
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    
    # create_dataset genera los tensores de entrenamiento y prueba
    dataset = create_dataset(f, n_var=2)
    return dataset


# 2. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO KAN

def entrenar_evaluar_kan():
    print("--- Iniciando preparación de datos ---")
    dataset = preparar_datos()

    # Crear modelo KAN: 

    print("--- Construyendo la red Kolmogorov-Arnold (KAN) ---")
    model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)

    print("--- Iniciando entrenamiento ---")
    start_time = time.time()
    
    # Entrenar el modelo usando el optimizador LBFGS (recomendado para KANs)
    results = model.fit(dataset, opt="LBFGS", steps=20, log=10)
    
    end_time = time.time()
    tiempo_entrenamiento = end_time - start_time
    
    # Extraer resultados
    train_loss = results['train_loss'][-1]
    test_loss = results['test_loss'][-1]

    # Imprimir resumen de la ejecución
    print(f"\n==========================================")
    print("RESUMEN DE RESULTADOS KAN")
    print("==========================================")
    print(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.4f} segundos")
    print(f"Pérdida en entrenamiento (RMSE): {train_loss:.6f}")
    print(f"Pérdida en prueba (RMSE): {test_loss:.6f}")

if __name__ == "__main__":
    entrenar_evaluar_kan()