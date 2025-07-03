# Отчет по Домашнему Заданию №3: Полносвязные сети

Студент: **Маслов Андрей Анатольевич**

Дата выполнения: **02 июля 2025**

Среда выполнения: **macOS (Apple M2), VS Code**

## Цель задания
Изучить влияние архитектуры полносвязных сетей на качество классификации, провести эксперименты с различными конфигурациями моделей.

## Задание 1: Эксперименты с глубиной сети (homework_depth_experiments.py)

В этом задании были сравнены модели с разным кол-вом слоев для анализа влияния глубины на точность и переобучение.

### Результат:
![Графики для модели depth_1_layer](result_screenshots/depth_results.png)
Подробные выводы в папке result_screenshots

### Анализ и графики:

**Оптимальная глубина:** Наилучшая точность (97.95%) получилась на модели с 3-мя слоями. Дальнейшее увеличение глубины без регуляризации привело к ухудшению результата.

**Переобучение:** С увеличением глубины выросла разница между точностью на обучающей и тестовой выборках, это признак переобучения.

**Регуляризация:** После добавления слоев BatchNorm и Dropout в 5-слойную модель получилось побороть переобучение и улучшить итоговую точность по сравнению с 5-слойной без регуляризации.

| Модель с 1 слоем (Линейная модель) | Модель с 2 слоями (Значитель прирост) |
| :---: | :---: |
| ![Графики для модели depth_1_layer](plots/depth_experiments/depth_1_layer_learning_curves.png) | ![Графики для модели depth_2_layers](plots/depth_experiments/depth_2_layers_learning_curves.png) |

| Модель с 3 слоем (Оптимальная глубина) | Модель с 5 слоями (Избыточная глубина) |
| :---: | :---: |
| ![Графики для модели depth_3_layer](plots/depth_experiments/depth_3_layers_learning_curves.png) | ![Графики для модели depth_5_layers](plots/depth_experiments/depth_5_layers_learning_curves.png) |

| Модель с 7 слоями (Переобучение) | 5-слойная модель с регуляризацией |
| :---: | :---: |
| ![Графики для модели depth_7_layers](plots/depth_experiments/depth_7_layers_learning_curves.png) | ![Графики для модели depth_5_layers_regularized](plots/depth_experiments/depth_5_layers_regularized_learning_curves.png) |


## Задание 2: Эксперименты с шириной сети (homework_width_experiments.py)

В этом задании я исследовал влияние кол-ва нейронов в слоях (ширины) на производительность и сложность модели.

### 2.1 Сравнение моделей разной ширины

### Результат:
![Результаты](result_screenshots/width_results.png)
Подробные выводы в папке result_screenshots

### Анализ и графики:

**Точность vs Затраты:** Увеличение ширины стабильно увеличивает точность, но ценой экспоненциального роста числа параметров и времени обучения. Самая широкая модель - very_wide показала отличный результат (98.22%), но она слишком дорога.

**Архитектурные схемы:** Эксперименты с сужающейся, расширяющейся и постоянной по ширине архитектурами показали очень близкие результаты, это означает меньшую важность конкретной схемы по сравнению с общей мощностью сети.

| Узкая модель (narrow) | Средняя модель (medium) |
| :---: | :---: |
| ![Графики для модели width_narrow](plots/width_experiments/width_narrow_learning_curves.png) | ![Графики для модели width_medium](plots/width_experiments/width_medium_learning_curves.png) |

| Широкая модель (wide) | Очень широкая модель (very_wide) |
| :---: | :---: |
| ![Графики для модели width_wide](plots/width_experiments/width_wide_learning_curves.png) | ![Графики для модели very_wide](plots/width_experiments/width_very_wide_learning_curves.png) |

### 2.2 Оптимизация архитектуры
В этом эксперименте сравнивались три разные схемы изменения ширины слоев.

### Результат:
| grid_constant |
| :---: |
| ![модель grid_constant](result_screenshots/grid_constant.png) |

| grid_expanding |
| :---: | 
| ![модель grid_expanding](result_screenshots/grid_expanding.png) |

| grid_contracting |
| :---: |
| ![модель grid_contracting](result_screenshots/grid_contracting.png) |

### Анализ и графики:

Все три схемы показали близкие результаты точности около 97.9%. Это говорит о том, что для данной задачи мощность сети оказалась важнее, чем конкретная схема их распределения по слоям.

| grid_constant |
| :---: | 
| ![Графики для модели grid_constant](plots/width_experiments/grid_constant_learning_curves.png) | 

| grid_expanding |
| :---: |
| ![Графики для модели grid_expanding](plots/width_experiments/grid_expanding_learning_curves.png) |

| grid_contracting |
| :---: |
| ![Графики для модели grid_contracting](plots/width_experiments/grid_contracting_learning_curves.png) |

## Задание 3: Эксперименты с регуляризацией (homework_regularization_experiments.py)

В этоq задаче сравнивали различные техники регуляризации на одной и той же базовой архитектуре.

### Результат:
![Результаты](result_screenshots/reg_results.png)
Подробные выводы в папке result_screenshots

### Анализ и графики

**Эффективность:** Абсолютно все техники регуляризации улучшили результат по сравнению с базовой моделью.

**Борьба с переобучением:** На всех графиках с регуляризацией видно, что кривые обучения для train и test выборок находятся гораздо ближе друг к другу, чем у модели reg_none.

**Сила комбинации:** Наилучший результат из всех проведенных экспериментов - 98.62%  показала модель с комбинацией Dropout и BatchNorm. Грамотное сочетание техник регуляризации эффективнее, чем простое наращивание размеров сети.

| Без регуляризации | Dropout (rate=0.1) |
| :---: | :---: |
| ![Графики для модели reg_none](plots/regularization_experiments/reg_none_learning_curves.png) | ![Графики для модели reg_dropout_0.1](plots/regularization_experiments/reg_dropout_0.1_learning_curves.png) |

| Dropout (rate=0.3) | Dropout (rate=0.5) |
| :---: | :---: |
| ![Графики для модели reg_dropout_0.3](plots/regularization_experiments/reg_dropout_0.3_learning_curves.png) | ![Графики для модели reg_dropout_0.5](plots/regularization_experiments/reg_dropout_0.5_learning_curves.png) |

| BatchNorm | Dropout + BatchNorm (Лучшая модель) |
| :---: | :---: |
| ![Графики для модели reg_batchnorm](plots/regularization_experiments/reg_batchnorm_learning_curves.png) | ![Графики для модели reg_dropout_batchnorm](plots/regularization_experiments/reg_dropout_batchnorm_learning_curves.png) |

| L2 |
| :---: |
| ![Графики для модели reg_batchnorm](plots/regularization_experiments/reg_l2_learning_curves.png) |

## Общий вывод
В ходе выполнения работы были успешно решены все поставленные задачи. Анализ полученных данных и графиков обучения наглядно продемонстрировал ключевые аспекты построения нейронных сетей: эксперименты с глубиной показали наличие оптимального количества слоев и риск переобучения при избыточной сложности; опыты с шириной выявили компромисс между мощностью модели и вычислительными затратами; а сравнение техник регуляризации доказало, что их грамотное применение, особенно комбинации BatchNorm и Dropout, является наиболее эффективным путем к созданию точных и стабильных моделей.
