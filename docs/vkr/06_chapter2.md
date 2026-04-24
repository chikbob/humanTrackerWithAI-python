# 2. Проектирование системы мониторинга и интеллектуального анализа

После анализа предметной области и формулирования требований необходимо определить архитектурные решения, обеспечивающие практическую реализацию системы. В данной главе рассматриваются общая архитектура приложения, его модульная структура, эксплуатационные режимы и ключевые пользовательские сценарии.

## 2.1. Разработка общей архитектуры системы

Фактическая архитектура проекта представляет собой многослойную Python-систему, в которой разделены пользовательский контур, фоновая обработка видеопотоков, сервисная логика, хранение данных и аналитические представления. Точкой входа интерфейса является `app.py`, а точкой входа фоновой обработки — `run_worker.py`.

Такое разделение принципиально: если бы интерфейс был владельцем production-видеопотока, обработка зависела бы от пользовательской сессии. Выделенный worker устраняет эту зависимость: он читает активные источники из базы, получает кадры, формирует события, сохраняет snapshots и обновляет `worker_status`, тогда как интерфейс работает с уже накопленными данными.

Слой компьютерного зрения распределен между каталогами `core/` и `video/`. В `core/` сосредоточены загрузка модели, работа с ROI и интерактивные сценарии, в `video/` — production-контур, включающий worker, pipeline и snapshots. `services/` отвечает за прикладную интерпретацию результатов обработки, `db/` — за хранение данных в `SQLite`, `analytics/` — за агрегированные представления, `config/` — за системные параметры, а `models/` — за базовые описания сущностей.

Поток данных имеет однозначное направление: источник передает кадры в worker или foreground-контур, далее выполняются детекция, трекинг и проверка ROI, после чего события, состояния worker и snapshots сохраняются в базе и runtime-каталоге, а интерфейс считывает эти данные и отображает пользователю. В архитектуре различаются `production path` и `demo/fallback path`, что позволяет совместить устойчивость и демонстрационную гибкость.

Рисунок 2.1 – Архитектурная схема веб-системы мониторинга и интеллектуального анализа объектов в реальном времени

Диаграмма отражает разделение системы на интерфейсный, фоновый, сервисный и аналитический контуры. В ней показано, что production-обработка выполняется отдельным worker-процессом, тогда как Streamlit-интерфейс работает с уже накопленными данными и состояниями.

```plantuml
@startuml
actor "Оператор" as Operator

node "Web UI" {
  component "app.py\nStreamlit entrypoint" as App
  package "ui/" {
    [Dashboard]
    [Online Monitoring]
    [Journal]
    [Analytics Views]
    [Employees]
    [Sources]
    [Settings]
  }
}

node "Background Processing" {
  component "run_worker.py" as RunWorker
  component "video/worker.py" as Worker
  component "video/pipeline.py" as Pipeline
  package "core/" {
    [Detection & Tracking]
  }
  package "services/" {
    [Events]
    [State]
    [EmployeeRepository]
    [Employee Sync]
    [Identity Placeholder]
  }
}

database "SQLite\nmonitoring.db" as DB
folder "runtime_data/\nsnapshots" as Runtime

package "analytics/" {
  [Access Analytics]
}

package "config/" {
  [App Config]
  [RTC Config]
}

package "models/" {
  [Entities]
}

cloud "Production video sources\nRTSP / HLS / USB" as ProdSources
cloud "Demo/Fallback sources\nBrowser / Local / Files" as DemoSources

Operator --> App
App --> Dashboard
App --> "Online Monitoring"
App --> Journal
App --> "Analytics Views"
App --> Employees
App --> Sources
App --> Settings

ProdSources --> Worker
RunWorker --> Worker
Worker --> Pipeline
Pipeline --> "Detection & Tracking"
Pipeline --> Events
Worker --> State
Worker --> DB
Worker --> Runtime
Worker --> "App Config"

DemoSources --> "Online Monitoring"
"Online Monitoring" --> "Detection & Tracking"
"Online Monitoring" --> Events
"Online Monitoring" --> State
"Online Monitoring" --> DB

App --> DB
App --> Runtime
App --> "Access Analytics"
App --> "EmployeeRepository"
App --> "Employee Sync"
App --> "RTC Config"
App --> Entities
@enduml
```

## 2.2. Проектирование модульной структуры приложения

Модульная структура проекта отражает реальные пользовательские и технологические контуры. К предметным модулям относятся dashboard, online monitoring, employees, event journal и video sources. К аналитическим — `analytics/access.py` и интерфейсные представления аналитики.

Инфраструктурную основу образуют worker, event generation, `EmployeeRepository`, database layer, механизмы `worker_status` и snapshots, конфигурационный слой и модели данных. Отдельную группу составляют демонстрационные компоненты: browser live, local camera, file monitoring и `live-window`.

Такая декомпозиция отделяет предметные модули от вычислительного и эксплуатационного контура и сохраняет прозрачность архитектуры.

Рисунок 2.2 – Компонентная структура веб-системы мониторинга и интеллектуального анализа объектов

На диаграмме показана декомпозиция приложения на предметные, аналитические, инфраструктурные и демонстрационные модули. Такая структура соответствует реальному разбиению проекта по каталогам и ролям компонентов.

```plantuml
@startuml
title Компонентная структура веб-системы мониторинга

package "UI / Предметные модули" {
  [Dashboard]
  [Online Monitoring]
  [Employees]
  [Event Journal]
  [Video Sources]
  [Settings]
}

package "Аналитические модули" {
  [Analytics Views]
  [Access Analytics]
}

package "Инфраструктурные модули" {
  [Worker]
  [Event Generation]
  [EmployeeRepository]
  [Database Layer]
  [Worker Status & Snapshots]
  [Config]
  [Models]
}

package "Демонстрационный контур" {
  [Browser Live]
  [Local Camera]
  [File Monitoring]
  [Live Window]
}

[Dashboard] --> [Access Analytics]
[Online Monitoring] --> [Event Generation]
[Online Monitoring] --> [Database Layer]
[Online Monitoring] --> [Live Window]
[Employees] --> [EmployeeRepository]
[Event Journal] --> [Database Layer]
[Event Journal] --> [EmployeeRepository]
[Video Sources] --> [Database Layer]
[Settings] --> [Config]
[Settings] --> [Database Layer]
[Analytics Views] --> [Access Analytics]
[Access Analytics] --> [Database Layer]

[Worker] --> [Event Generation]
[Worker] --> [Database Layer]
[Worker] --> [Worker Status & Snapshots]
[Worker] --> [Config]

[EmployeeRepository] --> [Database Layer]
[Models] --> [Database Layer]

[Browser Live] --> [Online Monitoring]
[Local Camera] --> [Online Monitoring]
[File Monitoring] --> [Online Monitoring]
[Live Window] --> [Online Monitoring]

note right of [Worker]
Production path
end note

note bottom of [Browser Live]
Demo/fallback path
end note

@enduml
```

## 2.3. Эксплуатационные режимы и технологические контуры системы

В текущем проекте корректнее говорить не о формализованной корпоративной ролевой модели, а об эксплуатационных режимах. Первый из них — операторский режим наблюдения, охватывающий dashboard, online monitoring, журнал событий и аналитику.

Второй режим связан с конфигурацией и источниками: пользователь добавляет и редактирует video sources, управляет их активностью и изменяет параметры обработки. Третий режим относится к справочнику сотрудников и статусу синхронизации employee directory.

Отдельно существует автоматизированный технологический контур — фоновый worker. Он не является пользовательской ролью, но выполняет постоянную production-обработку активных источников, формирует события и поддерживает `worker_status`. Интерфейс отвечает за представление и управление, worker — за непрерывную обработку, конфигурационный слой — за согласованность параметров.

## 2.4. Проектирование пользовательских сценариев и интерфейса

Пользовательские сценарии системы выстроены вокруг типового цикла работы оператора. Начальной точкой служит дашборд, после которого пользователь переходит к monitoring, журналу, аналитике, разделу сотрудников, источникам видео или настройкам.

Сценарий управления источниками обеспечивает добавление, редактирование, тестирование и активацию video sources. Сопровождение worker реализовано через просмотр `worker_status`, времени последнего кадра, heartbeat, ошибок и snapshots.

Центральным прикладным сценарием остается наблюдение за входной зоной. В online monitoring доступны production-источники, snapshots и interactive-сценарии. Интерфейс поддерживает multi-source monitoring с режимами «Фокус», «Сетка 2x2», «Список» и «Авто-компоновка», а `live-window` позволяет вынести выбранный источник в отдельное окно.

Отдельные сценарии связаны с журналом событий, аналитикой и справочником сотрудников. Журнал поддерживает фильтрацию, карточку события, ручную привязку к сотруднику и экспорт выборки. Аналитический раздел предоставляет агрегаты по времени, типам событий, точкам доступа и состоянию источников. Раздел сотрудников поддерживает локальный и удаленный режимы работы со справочником.

Интерфейс также должен честно различать `production monitoring` и `demo/fallback` режимы. Первое опирается на server-side обработку worker, второе — на browser live, локальную камеру и файловые сценарии в пределах пользовательской сессии.

Рисунок 2.3 – Диаграмма вариантов использования веб-системы мониторинга и интеллектуального анализа объектов

Диаграмма фиксирует основные действия пользователя и автономные функции фонового worker. Она показывает, какие сценарии доступны через интерфейс и какие процессы система выполняет без участия оператора.

```plantuml
@startuml
left to right direction
actor "Пользователь системы" as User
actor "Фоновый worker" as Worker

rectangle "Веб-система мониторинга" {
  usecase "Просмотр дашборда" as UC1
  usecase "Выбор и активация\nисточников видео" as UC2
  usecase "Контроль состояния\nисточников" as UC3
  usecase "Наблюдение за входной зоной" as UC4
  usecase "Работа с multi-source\nmonitoring" as UC5
  usecase "Выбор режима отображения\nФокус / Сетка 2x2 /\nСписок / Авто-компоновка" as UC6
  usecase "Открытие источника\nв live-window" as UC7
  usecase "Просмотр snapshot и\nlive-состояния" as UC8
  usecase "Просмотр и фильтрация\nжурнала событий" as UC9
  usecase "Ручная привязка события\nк сотруднику" as UC10
  usecase "Работа со справочником\nсотрудников" as UC11
  usecase "Запуск синхронизации\nemployee directory" as UC12
  usecase "Просмотр аналитики" as UC13
  usecase "Изменение системных\nнастроек" as UC14

  usecase "Непрерывная обработка\nproduction-источников" as UC15
  usecase "Формирование событий\nпроходной" as UC16
  usecase "Обновление worker_status\nи snapshots" as UC17
}

User --> UC1
User --> UC2
User --> UC3
User --> UC4
User --> UC5
User --> UC6
User --> UC7
User --> UC8
User --> UC9
User --> UC10
User --> UC11
User --> UC12
User --> UC13
User --> UC14

Worker --> UC15
Worker --> UC16
Worker --> UC17

UC4 ..> UC8 : <<include>>
UC5 ..> UC6 : <<include>>
UC9 ..> UC10 : <<extend>>
UC11 ..> UC12 : <<extend>>
UC15 ..> UC16 : <<include>>
UC15 ..> UC17 : <<include>>

@enduml
```
