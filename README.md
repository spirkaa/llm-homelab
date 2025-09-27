# LLM HomeLab

Конфигурация инструментов для локального запуска языковых моделей.

## Команды

### Загрузка-обновление-сборка образов и запуск всех контейнеров

1. Запустить

    ```shell
    ./run.sh
    ```

### Загрузка моделей

1. Скопировать .envrc

    ```shell
    cp .envrc.example .envrc
    ```

2. Указать токен [Hugging Face](https://huggingface.co/settings/tokens) в переменной `HF_TOKEN` в файле .envrc
3. Разрешить чтение .envrc

    ```shell
    direnv allow
    ```

4. Указать ссылку на модель в переменной URL в файле download_models.sh
5. Запустить

    ```shell
    ./download_models.sh
    ```

## Железо

* MSI GeForce RTX 3090 Gaming X Trio 24G
* AMD Ryzen 7 3800X
* GIGABYTE X570 AORUS PRO
* 4x16 GB DDR4-3200 G.SKILL F4-3200C16S-16GVK
* 1 TB Samsung 980 Pro
* Fractal Design Define R6, Corsair RM850i
