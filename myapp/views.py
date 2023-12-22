from django.shortcuts import render,redirect
from django.http import HttpResponse, JsonResponse
from .models import Project
from .models import Task
from django.shortcuts import render
from .forms import CreateNewTask
from PIL import Image
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
from django.conf import settings
# Create your views here.
# En helpers.py
from PIL import Image
import os
from django.conf import settings
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from IPython.display import display
from myapp.django_requirements import detect_foliage_state
from sklearn.metrics import accuracy_score
from minisom import MiniSom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json

def convertir_a_escala_de_grises(imagen, nombre_archivo):
    # Verificar si la carpeta de destino existe, y crearla si no
    carpeta_destino = os.path.join(settings.STATICFILES_DIRS[0])
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Abrir la imagen
    img = Image.open(imagen)

    # Convertir a escala de grises
    img_gris = img.convert("L")

    # Guardar la imagen en una carpeta de archivos estáticos
    ruta_destino = os.path.join(carpeta_destino, nombre_archivo)
    img_gris.save(ruta_destino, format="PNG")

    return nombre_archivo

def filtrarImagenes(image_path):
    carpeta_destino = os.path.join(settings.STATICFILES_DIRS[0])
    with open(carpeta_destino+'/matrices_orientacion1.json', 'r') as json_file:
     data = json.load(json_file)

    # Reformatear los datos y agregar etiquetas (sana: 0, enferma: 1)
    X = np.array(data).reshape(len(data), -1)
    y = np.array([i % 3 for i in range(len(data))])

    # Normalizar los datos
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Definir y entrenar la red neuronal SOM
    som_rows, som_cols = 5, 5  # Ajustar según la cantidad deseada de neuronas en el SOM
    som = MiniSom(som_rows, som_cols, X_train.shape[1], sigma=0.5, learning_rate=0.5)
    # Modificación para mostrar las iteraciones
    for epoch in range(1):  # 3000 iteraciones de entrenamiento
        som.train_random(X_train, 1)  # Entrenar con una muestra aleatoria
        if epoch % 1 == 0:
            print(f"Iteración {epoch}/{3000} ------------")

    # Asignar etiquetas a las neuronas ganadoras
    winners = np.array([som.winner(x) for x in X_train])
    labels_map = np.zeros((som_rows, som_cols), dtype=int)
    for i, x in enumerate(winners):
        labels_map[x] += y_train[i]



    # Asignar la etiqueta mayoritaria a cada neurona
    predicted_labels = np.array([labels_map[x].argmax() for x in winners])

    # Calcular la precisión del modelo en el conjunto de entrenamiento
    accuracy_train = accuracy_score(y_train, predicted_labels)
    print(f'Accuracy en el conjunto de entrenamiento: {accuracy_train:.2%}')

    # Calcular la precisión del modelo en el conjunto de prueba
    winners_test = np.array([som.winner(x) for x in X_test])
    predicted_labels_test = np.array([labels_map[x].argmax() for x in winners_test])
    accuracy_test = accuracy_score(y_test, predicted_labels_test)
    # print(f'Accuracy en el conjunto de prueba: {accuracy_test:.2%}')


    def probar(image_path):
        carpeta_destino = os.path.join(settings.STATICFILES_DIRS[0])

        # Función para procesar una imagen individual
        def procesar_imagen(imagen, dimensiones_comunes=(25, 10)):
            ruta_imagen_local = carpeta_destino+'/imagen_inicial.jpg'

            # Lee la imagen localmente
            imagen = cv2.imread(ruta_imagen_local)

            cv2.imwrite(carpeta_destino+'/img_original.png', imagen)
            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            # Aplicar un desenfoque para mejorar la detección de bordes
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Aplicar la detección de bordes usando Canny
            edges = cv2.Canny(blurred, 50, 150)

            gradiente_x, gradiente_y = np.gradient(edges)
            # Encontrar contornos en la imagen
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Encontrar el contorno con el área máxima
            max_contour = max(contours, key=cv2.contourArea)

            # Calcular el rectángulo delimitador para el contorno
            x, y, w, h = cv2.boundingRect(max_contour)

            # Recortar la región de interés (ROI) de la imagen original
            cropped_image = imagen[y:y+h, x:x+w]

            # Mostrar la imagen original y la imagen recortada
            plt.figure(figsize=(15, 3))
            # plt.subplot(151), plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)), plt.title('Imagen \nOriginal')
            # plt.subplot(152), plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)), plt.title('Imagen Recortada')


            # Guardar la imagen recortada
  

            cv2.imwrite(carpeta_destino+'/img_recortada.png', cropped_image)

            # Aplicar transformadas de Fourier, filtrar frecuencias y mostrar resultados
            aplicar_fft_y_recortar(carpeta_destino+'/img_recortada.png')

            matriz_orientacion = calcular_orientaciones_promedio(gradiente_x, gradiente_y, dimensiones_comunes[0], dimensiones_comunes[1])
            return matriz_orientacion



        # Función de orientación con ajuste de proporciones
        def calcular_orientaciones_promedio(Gx, Gy, ancho_region=18, alto_region=18, dimensiones_comunes=(25, 10)):
            m, n = Gx.shape
            target_rows, target_cols = dimensiones_comunes
            matriz_orientacion = np.zeros((target_rows, target_cols))

            for i in range(0, target_rows * alto_region, alto_region):
                for j in range(0, target_cols * ancho_region, ancho_region):
                    # Extraer la región de tamaño ancho_region x alto_region
                    region_Gx = Gx[i:i + alto_region, j:j + ancho_region]
                    region_Gy = Gy[i:i + alto_region, j:j + ancho_region]

                    # Calcular la orientación promedio para la región
                    orientacion_promedio = np.arctan2(np.mean(region_Gy), np.mean(region_Gx))

                    # Si la región tiene una magnitud promedio cercana a cero, asignar orientación a cero
                    if np.sqrt(np.mean(region_Gx)**2 + np.mean(region_Gy)**2) < 1e-5:
                        orientacion_promedio = 0.0

                    # Calcular los índices para la matriz de orientación
                    indice_fila = i // alto_region
                    indice_columna = j // ancho_region

                    # Actualizar la matriz de orientación con la orientación promedio
                    matriz_orientacion[indice_fila, indice_columna] = orientacion_promedio
        
            # Configurar valores "nan" como cero
            matriz_orientacion = np.nan_to_num(matriz_orientacion, nan=0)
            return matriz_orientacion


        def aplicar_fft_y_recortar(imagen_path):
            imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE) # Cargar la imagen desde el archivo
            imagen_fft = fft2(imagen) # Aplicar FFT, descomponer en sus componentes de frecuencia

            # Mover el componente de baja frecuencia al centro
            imagen_fft_shifted = fftshift(imagen_fft)

            # Eliminar frecuencias bajas y altas (filtro de paso de banda)
            rows, cols = imagen.shape
            centro_row, centro_col = rows // 2 , cols // 2
            radio_paso_banda = 5
            imagen_fft_shifted[centro_row - radio_paso_banda:centro_row + radio_paso_banda,
                            centro_col - radio_paso_banda:centro_col + radio_paso_banda] = 0

            # Aplicar la transformada inversa para obtener la imagen filtrada
            imagen_filtrada = np.abs(ifft2(fftshift(imagen_fft_shifted)))

            umbral = 80

            # Aplicar umbral para obtener una máscara binaria
            _, mascara = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)  # Aplicar umbral a la imagen original

            # Aplicar la máscara a la imagen en escala de grises antes de la transformada inversa de Fourier
            imagen_con_mascara = cv2.bitwise_and(imagen, imagen, mask=mascara)

            # Aplicar el operador Canny
            bordes_canny = cv2.Canny(imagen_con_mascara, 50, 150)

            # Calcular los gradientes
            gradiente_x, gradiente_y = np.gradient(bordes_canny)

            # Calcular la matriz de orientación con proporciones 18x18 directamente desde el gradiente
            matriz_orientacion = calcular_orientaciones_promedio(gradiente_x, gradiente_y, 18, 18)


            # Guardar cada imagen sin bordes ni texto
            output_path_original_gray = carpeta_destino+'/imagen_original_gray.png'
            output_path_filtrada = carpeta_destino+'/imagen_filtrada.png'
            #output_path_orientacion = 'static/myapp/grises/matriz_orientacion.png'

            cv2.imwrite(output_path_original_gray, imagen)  # Guardar imagen original
            cv2.imwrite(output_path_filtrada, imagen_filtrada)  # Guardar imagen filtrada
            #plt.imsave(output_path_orientacion, matriz_orientacion, cmap='hsv')  # Guardar matriz de orientación

            print("Matriz de Orientación:")
            # display(matriz_orientacion)

            # Mostrar la imagen original, filtrada y la matriz de orientación
            #plt.subplot(153), plt.imshow(imagen, cmap='gray'), plt.title('Hoja')
            #plt.subplot(154), plt.imshow(imagen_filtrada, cmap='gray'), plt.title('Filtrada con FFT')
            #plt.subplot(155), plt.imshow(matriz_orientacion, cmap='hsv'), plt.title('Matriz de Orientación')
            plt.subplot(155), plt.imshow(matriz_orientacion, cmap='hsv'), plt.axis('off')
            plt.imshow(matriz_orientacion, cmap='hsv')

            plt.savefig(carpeta_destino+'/matriz_orientacion.png', bbox_inches='tight', pad_inches=0)
        

            return imagen_filtrada
        
        img = cv2.imread(image_path)
        # Paso 1: Preprocesamiento y normalización
        nueva_matriz_orientacion = procesar_imagen(img, dimensiones_comunes=(25, 10))  # Obtén la matriz de orientación de la nueva imagen
        nueva_matriz_orientacion = np.array(nueva_matriz_orientacion).flatten().reshape(1, -1)
        nueva_matriz_orientacion_normalizada = scaler.transform(nueva_matriz_orientacion)


        # Paso 2: Mapeo en el SOM
        neurona_ganadora = som.winner(nueva_matriz_orientacion_normalizada)

        
        # Paso 3: Asignación de Etiqueta a la Neurona Ganadora
        etiqueta_predicha = labels_map[neurona_ganadora].argmax()
        etiqueta_pred = str(etiqueta_predicha)

        # Paso 4: Clasificación Final
        if etiqueta_pred == "0.":
            print("La hoja está sana.")

        elif etiqueta_pred == "1.":
            print("La hoja está enferma A.")

        
        elif etiqueta_pred == "2.":
            print("La hoja está enferma B.")
        
        detect_state = detect_foliage_state(image_path)
        return detect_state


        
        

    return probar(image_path)

   

def guardarStaticImg(imagenPost):
    carpeta_destino = os.path.join(settings.STATICFILES_DIRS[0])
    img = Image.open(imagenPost)
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    ruta_destino = os.path.join(carpeta_destino, 'imagen_inicial.jpg')
    img.save(ruta_destino, format="PNG")


def index(request):
    title = 'Django Course!!'
    
    # Procesar el formulario si se ha enviado
    if request.method == 'POST' and request.FILES.get('imagen'):
        imagen_original = request.FILES['imagen']
        print(type(imagen_original))
        guardarStaticImg(imagen_original)
        # Generar un nombre único para la imagen
        nombre_archivo = 'imagen_gris.png'  # Puedes utilizar algún método para generar un nombre único

        # Llamar a la función para convertir y guardar la imagen
        #convertir_a_escala_de_grises(imagen_original, nombre_archivo)
        carpeta_destino = os.path.join(settings.STATICFILES_DIRS[0])
        image_path = carpeta_destino+'/'+str(imagen_original)
        precision_tipoHoja = filtrarImagenes(image_path)
       
    else:
        nombre_archivo = None
        # nombre_archivo = 'wqe'
        precision_tipoHoja = ''

    return render(request, 'index.html', {
        'title': title,
        'nombre_archivo': nombre_archivo,
        'precision_tipoHoja':precision_tipoHoja
    })

def hello(request):
            # Crear la respuesta JSON
    response_data = {'message': 'Hello, World!'}
    
    # Enviar la respuesta JSON
    return JsonResponse(response_data)

def about(request):
    username = 'ALfonso contreras'
    return render(request,'about.html',{
        'username':username
    })

def projects(request):
    #projects = list(Project.objects.values())
    projects = Project.objects.all()
    #return JsonResponse(projects, safe=False)
    return render(request,'projects.html',{
        'projects':projects
    })

def tasks(request):
    #task = Task.objects.get(title=title)
    #return HttpResponse('task: %s'%task.title)
    tasks = Task.objects.all()
    return render(request,'task.html',{
        'tasks':tasks
    })

def create_task(request):
    if request.method == 'GET':
        return render(request, 'create_task.html',{
            'form': CreateNewTask()
        })
    else:
        Task.objects.create(title=request.POST['title'],description=request.POST['description'],project_id=2)
        return redirect('/tasks/')
