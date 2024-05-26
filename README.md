# URFU_SE_project

## Использование проекта
1) Необходимо склонировать репозиторый в рабочую папку.

2) Запустить скрипт ```prepare.sh```, который создаст папку ```pretrained```, в которую скачаются необходимые для работы приложения веса предобученных моделей

## Создания датасета с эмбедингами людей
Для создания необходимо воспользоваться скриптом ```create_database.py```, которой принимает в качестве аргумента путь до папки с изображениями лиц людей.

Структура папки с лицами людей должна выглядить следующим образом: 

    |- <название папки с людьми>
        |- <имя первой персоны>
            |- далее идут n изображений с лицом конкретного человека
        |- ...
        |- <имя последней персоны>
            |- далее идут m изображений с лицом конкретного человека

После выполнения данного скрипта в рабочей директории появится папка ```embeddings_database```. Она будет иметь такую же структуру, как и папка с лицами, только вместо изображений будет находится один файл в npy формате с усредненным эмбедингом со всех изображений.

## Использование FastAPI

### Приложение, имитирующее работу системы повторной идентификации человека

Для запуска сервера необходимо ввести следующую команду в консоли:
```bash
python main.py
```
Заходим в браузере по адресу http://localhost:8000/
Откроется окно с интерфейсом сайта
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_reid_demo_image1.jpg"></a>
</div>
В данном окне имеется возможность выбрать одну из двух функций: Повторная идентификация человека по базе и добавление/обновление персоны в базе.

При нажатии кнопки ```Reid person``` переходим на страницу со следующим интерфейсом:
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_reid_demo_image2.jpg"></a>
</div>

Можно загрузить изображение и нажать кнопку ```Submit``` или вернуться на домашнюю страницу. После нажатия кнопки ```Submit``` в окне отобразится загруженное изображение, идентификатор персоны в базе и возвращение на домашнюю страницу. В случае если персоны нет в базе в строке Person id будет надпись ```Unknown```.
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_reid_demo_image3.jpg"></a>
</div>

При нажатии кнопки ```Add new person``` переходим на страницу со следующим интерфейсом:
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_reid_demo_image4.jpg"></a>
</div>

Можно загрузить изображение, ввести person id и нажать кнопку ```Save person``` или вернуться на домашнюю страницу. В случае успешного добавления появится следующая страница:
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_reid_demo_image5.jpg"></a>
</div>

В случае ошибки будет выведено сообщение ошибки 
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_reid_demo_image6.jpg"></a>
</div>

### Приложение для сравнения людей на двух изображаениях
Для запуска сервера необходимо ввести следующую команду в консоли:
```bash
python server_reid.py
```
Заходим в браузере по адресу http://localhost:8000/
Откроется окно с интерфейсом сайта
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_demo_image1.jpg"></a>
</div>
В данном окне можно загрузить два изображения для сравнения людей, нажав кнопку "выбрать файл".
После нажатия кнопки "Submit" на экране выведится два изображения с людьми и их косинусная похожесть.
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_demo_image2.jpg"></a>
</div>
<div align="center">
  <img width="100%" src="https://github.com/Dramb1/URFU_SE_project/blob/main/data/fastapi_demo_image3.jpg"></a>
</div>
