# maestria-trabajo-integrador

🐍 Configuración del Entorno Virtual
Para garantizar que las dependencias de este proyecto no interfieran con otras instalaciones de Python en tu sistema, es recomendable crear un entorno virtual. Sigue estos pasos según tu sistema operativo:

<br>1. Crear el Entorno Virtual</br>

En Windows:

python -m venv .venv

En macOS o Linux:

python3 -m venv .venv

Esto creará un directorio oculto .venv en el que se instalarán las dependencias del proyecto.

<br>2. Activar el Entorno Virtual</br>

En Windows:

.venv\Scripts\activate

En macOS o Linux:

source .venv/bin/activate

Una vez activado, el prompt de tu terminal debería mostrar el nombre del entorno entre paréntesis, indicando que está activo.

<br>3. Instalar las Dependencias</br>

Con el entorno virtual activado, instala las bibliotecas necesarias para el proyecto ejecutando:

pip install -r requirements.txt

Esto instalará todas las dependencias listadas en el archivo requirements.txt, asegurando que el entorno esté configurado correctamente.

<br>4. Desactivar el Entorno Virtual</br>

deactivate
