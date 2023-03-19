Устанавливаем зависимости:
```
python -m venv .venv --prompt sd
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python scripts/gradio_superres.py
```

Запускаем модель:

```
python run.py --path_1 "examples/1/1.png" --path_2 "examples/1/2.png" --save_path "examples/1/output.png"
```

где
--path_1 - путь к первому изображению
--path_2 - путь ко второму изображению
--save_path - путь (вместе с названием файла) сохранения выхода модели
