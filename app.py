from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import io
import base64
import os
from datetime import datetime, timedelta
import csv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'predictive-analytics-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///telemetry.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ──────────────────────────────────────────────
# МОДЕЛИ БД
# ──────────────────────────────────────────────

class Sensor(db.Model):
    __tablename__ = 'sensor'
    sensor_id = db.Column(db.Integer, primary_key=True)
    name      = db.Column(db.String(100), nullable=False)
    unit      = db.Column(db.String(50))
    description = db.Column(db.Text)
    telemetry = db.relationship('Telemetry',  backref='sensor', lazy=True, cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', backref='sensor', lazy=True, cascade='all, delete-orphan')

class Telemetry(db.Model):
    __tablename__ = 'telemetry'
    id        = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    sensor_id = db.Column(db.Integer, db.ForeignKey('sensor.sensor_id'), nullable=False)
    value     = db.Column(db.Float, nullable=False)

class Prediction(db.Model):
    __tablename__ = 'prediction'
    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp       = db.Column(db.DateTime, nullable=False)
    sensor_id       = db.Column(db.Integer, db.ForeignKey('sensor.sensor_id'), nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    model_name      = db.Column(db.String(50), default='LinearRegression')

with app.app_context():
    db.create_all()
    # Создаём датчики по умолчанию если их нет
    if Sensor.query.count() == 0:
        sensors = [
            Sensor(sensor_id=1, name='Температура',  unit='°C',     description='Датчик температуры оборудования'),
            Sensor(sensor_id=2, name='Вибрация',     unit='мм/с',   description='Датчик вибрации подшипников'),
            Sensor(sensor_id=3, name='Давление',     unit='атм',    description='Датчик давления в системе'),
        ]
        db.session.add_all(sensors)
        db.session.commit()

# ──────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────

SENSOR_COLORS = {1: '#3b82f6', 2: '#f59e0b', 3: '#10b981'}
SENSOR_LABELS = {1: 'Температура (°C)', 2: 'Вибрация (мм/с)', 3: 'Давление (атм)'}

def generate_demo_data():
    """Генерирует демо-данные для 3 датчиков с трендом и сезонностью."""
    np.random.seed(42)
    base_time = datetime.now() - timedelta(hours=16, minutes=40)
    records = []
    params = {
        1: dict(base=25.0, trend=0.05,  amplitude=7.0,  noise=1.5,  period=200),
        2: dict(base=3.0,  trend=0.01,  amplitude=1.5,  noise=0.3,  period=100),
        3: dict(base=5.0,  trend=0.008, amplitude=0.8,  noise=0.2,  period=150),
    }
    for i in range(200):
        ts = base_time + timedelta(minutes=5 * i)
        for sid, p in params.items():
            val = (p['base']
                   + p['trend'] * i
                   + p['amplitude'] * np.sin(2 * np.pi * i / p['period'])
                   + np.random.normal(0, p['noise']))
            records.append(Telemetry(timestamp=ts, sensor_id=sid, value=round(val, 4)))
    db.session.add_all(records)
    db.session.commit()
    return len(records)


def make_plot(timestamps, values, sensor_id, title='Телеметрия', pred_ts=None, pred_vals=None):
    """Строит matplotlib-график и возвращает base64-строку."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 4))
    color = SENSOR_COLORS.get(sensor_id, '#6366f1')

    ax.plot(timestamps, values, color=color, linewidth=1.8, label='Факт', zorder=3)

    if pred_ts and pred_vals:
        ax.plot(pred_ts, pred_vals, color='#ef4444', linewidth=2,
                linestyle='--', label='Прогноз', zorder=4)
        ax.axvline(x=timestamps[-1], color='#9ca3af', linestyle=':', linewidth=1.2)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Время', fontsize=10)
    ax.set_ylabel(SENSOR_LABELS.get(sensor_id, 'Значение'), fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def run_prediction(sensor_id, steps):
    """Линейная регрессия по данным датчика. Возвращает (pred_ts, pred_vals, r2)."""
    rows = (Telemetry.query
            .filter_by(sensor_id=sensor_id)
            .order_by(Telemetry.timestamp)
            .all())
    if len(rows) < 3:
        return None, None, None

    base_ts  = rows[0].timestamp
    X = np.array([(r.timestamp - base_ts).total_seconds() for r in rows]).reshape(-1, 1)
    y = np.array([r.value for r in rows])

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    last_sec   = X[-1, 0]
    step_sec   = (X[-1, 0] - X[0, 0]) / max(len(X) - 1, 1)
    future_X   = np.array([last_sec + step_sec * (i + 1) for i in range(steps)]).reshape(-1, 1)
    future_y   = model.predict(future_X)
    future_ts  = [rows[-1].timestamp + timedelta(seconds=step_sec * (i + 1)) for i in range(steps)]

    return future_ts, future_y.tolist(), round(r2, 4)

# ──────────────────────────────────────────────
# МАРШРУТЫ
# ──────────────────────────────────────────────

@app.route('/')
def index():
    sensors   = Sensor.query.all()
    total_rec = Telemetry.query.count()
    stats = []
    for s in sensors:
        rows = Telemetry.query.filter_by(sensor_id=s.sensor_id).all()
        if rows:
            vals = [r.value for r in rows]
            last = rows[-1]
            stats.append({
                'sensor': s,
                'count': len(rows),
                'last_value': round(last.value, 3),
                'last_ts': last.timestamp,
                'min_val': round(min(vals), 3),
                'max_val': round(max(vals), 3),
                'avg_val': round(sum(vals) / len(vals), 3),
            })
        else:
            stats.append({'sensor': s, 'count': 0, 'last_value': None,
                          'last_ts': None, 'min_val': None, 'max_val': None, 'avg_val': None})
    return render_template('index.html', stats=stats, total_rec=total_rec)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # ── Генерация демо-данных ──
        if 'generate_demo' in request.form:
            n = generate_demo_data()
            flash(f'✅ Сгенерировано {n} демо-записей для 3 датчиков.', 'success')
            return redirect(url_for('data_view'))

        # ── Загрузка CSV ──
        file = request.files.get('csv_file')
        if not file or file.filename == '':
            flash('⚠️ Выберите CSV-файл.', 'warning')
            return redirect(url_for('upload'))
        try:
            df = pd.read_csv(file)
            required = {'timestamp', 'sensor_id', 'value'}
            if not required.issubset(df.columns):
                flash(f'❌ CSV должен содержать колонки: {", ".join(required)}', 'danger')
                return redirect(url_for('upload'))

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            added = 0
            for _, row in df.iterrows():
                # Убедимся что датчик существует
                sid = int(row['sensor_id'])
                if not Sensor.query.get(sid):
                    db.session.add(Sensor(sensor_id=sid,
                                          name=f'Датчик {sid}',
                                          unit='—',
                                          description='Импортирован из CSV'))
                db.session.add(Telemetry(
                    timestamp=row['timestamp'].to_pydatetime(),
                    sensor_id=sid,
                    value=float(row['value'])
                ))
                added += 1
            db.session.commit()
            flash(f'✅ Загружено {added} записей из CSV.', 'success')
            return redirect(url_for('data_view'))
        except Exception as e:
            db.session.rollback()
            flash(f'❌ Ошибка при обработке файла: {e}', 'danger')
            return redirect(url_for('upload'))

    return render_template('upload.html')


@app.route('/manual', methods=['GET', 'POST'])
def manual():
    sensors = Sensor.query.all()
    if request.method == 'POST':
        try:
            ts  = datetime.strptime(request.form['timestamp'], '%Y-%m-%dT%H:%M')
            sid = int(request.form['sensor_id'])
            val = float(request.form['value'])
            db.session.add(Telemetry(timestamp=ts, sensor_id=sid, value=val))
            db.session.commit()
            sname = Sensor.query.get(sid).name if Sensor.query.get(sid) else sid
            flash(f'✅ Запись добавлена: датчик «{sname}», значение {val}.', 'success')
            return redirect(url_for('data_view'))
        except Exception as e:
            db.session.rollback()
            flash(f'❌ Ошибка: {e}', 'danger')
    return render_template('manual.html', sensors=sensors,
                           now=datetime.now().strftime('%Y-%m-%dT%H:%M'))


@app.route('/data')
def data_view():
    sensors    = Sensor.query.all()
    sensor_id  = request.args.get('sensor_id', type=int)

    query = Telemetry.query
    if sensor_id:
        query = query.filter_by(sensor_id=sensor_id)
    query = query.order_by(Telemetry.timestamp.desc())
    records = query.limit(100).all()

    # График
    plot_b64 = None
    if records:
        plot_records = list(reversed(records))
        sid_for_plot = sensor_id if sensor_id else (plot_records[0].sensor_id if plot_records else 1)
        ts_list  = [r.timestamp for r in plot_records]
        val_list = [r.value     for r in plot_records]
        sname    = Sensor.query.get(sid_for_plot).name if Sensor.query.get(sid_for_plot) else ''
        plot_b64 = make_plot(ts_list, val_list, sid_for_plot,
                             title=f'Телеметрия — {sname}')

    return render_template('data.html',
                           records=records,
                           sensors=sensors,
                           selected_sensor=sensor_id,
                           plot_b64=plot_b64,
                           total=Telemetry.query.count())


@app.route('/delete_all', methods=['POST'])
def delete_all():
    n = Telemetry.query.count()
    Telemetry.query.delete()
    db.session.commit()
    flash(f'🗑️ Удалено {n} записей из базы данных.', 'info')
    return redirect(url_for('data_view'))


@app.route('/export_data')
def export_data():
    rows = (Telemetry.query
            .join(Sensor, Telemetry.sensor_id == Sensor.sensor_id)
            .order_by(Telemetry.timestamp)
            .with_entities(Telemetry.timestamp, Sensor.name, Telemetry.sensor_id, Telemetry.value)
            .all())

    def generate():
        yield 'timestamp,sensor_name,sensor_id,value\n'
        for r in rows:
            yield f'{r.timestamp},{r.name},{r.sensor_id},{r.value}\n'

    return Response(generate(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=telemetry_export.csv'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    sensors    = Sensor.query.all()
    plot_b64   = None
    pred_table = []
    r2_score   = None
    selected_sensor_id = None
    steps      = 10

    if request.method == 'POST':
        try:
            selected_sensor_id = int(request.form['sensor_id'])
            steps = int(request.form.get('steps', 10))
            steps = max(1, min(steps, 200))

            rows = (Telemetry.query
                    .filter_by(sensor_id=selected_sensor_id)
                    .order_by(Telemetry.timestamp)
                    .all())

            if len(rows) < 3:
                flash('⚠️ Недостаточно данных для прогноза (нужно ≥ 3 записей).', 'warning')
            else:
                ts_list  = [r.timestamp for r in rows]
                val_list = [r.value     for r in rows]

                pred_ts, pred_vals, r2_score = run_prediction(selected_sensor_id, steps)

                sname    = Sensor.query.get(selected_sensor_id).name
                plot_b64 = make_plot(ts_list, val_list, selected_sensor_id,
                                     title=f'Прогноз линейной регрессии — {sname}',
                                     pred_ts=pred_ts, pred_vals=pred_vals)

                pred_table = [{'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                               'value': round(v, 4)}
                              for t, v in zip(pred_ts, pred_vals)]

        except Exception as e:
            flash(f'❌ Ошибка прогнозирования: {e}', 'danger')

    return render_template('predict.html',
                           sensors=sensors,
                           plot_b64=plot_b64,
                           pred_table=pred_table,
                           r2_score=r2_score,
                           selected_sensor_id=selected_sensor_id,
                           steps=steps)


@app.route('/export_prediction')
def export_prediction():
    sensor_id = request.args.get('sensor_id', type=int)
    steps     = request.args.get('steps', 10, type=int)

    pred_ts, pred_vals, _ = run_prediction(sensor_id, steps)
    if pred_ts is None:
        flash('⚠️ Нет данных для экспорта прогноза.', 'warning')
        return redirect(url_for('predict'))

    sname = Sensor.query.get(sensor_id).name if Sensor.query.get(sensor_id) else sensor_id

    def generate():
        yield 'timestamp,sensor_name,predicted_value,model\n'
        for t, v in zip(pred_ts, pred_vals):
            yield f'{t.strftime("%Y-%m-%d %H:%M:%S")},{sname},{round(v, 4)},LinearRegression\n'

    return Response(generate(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename=prediction_sensor{sensor_id}.csv'})

# ══════════════════════════════════════════════
#  МЕНЕДЖЕР БАЗЫ ДАННЫХ
# ══════════════════════════════════════════════

# ── Главная страница менеджера ──
@app.route('/db')
def db_manager():
    # Датчики
    sensors = Sensor.query.order_by(Sensor.sensor_id).all()

    # Телеметрия с пагинацией
    page     = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    search_sid = request.args.get('search_sid', '', type=str)

    query = Telemetry.query
    if search_sid.strip():
        try:
            query = query.filter_by(sensor_id=int(search_sid))
        except ValueError:
            pass

    pagination = (query
                  .order_by(Telemetry.timestamp.desc())
                  .paginate(page=page, per_page=per_page, error_out=False))

    # Статистика таблиц
    db_stats = {
        'sensors_count':   Sensor.query.count(),
        'telemetry_count': Telemetry.query.count(),
        'prediction_count': Prediction.query.count(),
    }

    return render_template('db_manager.html',
                           sensors=sensors,
                           pagination=pagination,
                           db_stats=db_stats,
                           search_sid=search_sid,
                           per_page=per_page)


# ══ CRUD ДАТЧИКИ ══

@app.route('/db/sensor/add', methods=['GET', 'POST'])
def sensor_add():
    if request.method == 'POST':
        try:
            sid  = request.form.get('sensor_id', '').strip()
            name = request.form['name'].strip()
            unit = request.form.get('unit', '').strip()
            desc = request.form.get('description', '').strip()

            if not name:
                flash('❌ Название датчика обязательно.', 'danger')
                return redirect(url_for('sensor_add'))

            # Авто-ID если не задан
            if sid:
                sid = int(sid)
                if Sensor.query.get(sid):
                    flash(f'❌ Датчик с ID {sid} уже существует.', 'danger')
                    return redirect(url_for('sensor_add'))
            else:
                max_id = db.session.query(db.func.max(Sensor.sensor_id)).scalar() or 0
                sid = max_id + 1

            db.session.add(Sensor(sensor_id=sid, name=name, unit=unit, description=desc))
            db.session.commit()
            flash(f'✅ Датчик #{sid} «{name}» успешно добавлен.', 'success')
            return redirect(url_for('db_manager'))
        except Exception as e:
            db.session.rollback()
            flash(f'❌ Ошибка: {e}', 'danger')

    # Предложить следующий свободный ID
    max_id = db.session.query(db.func.max(Sensor.sensor_id)).scalar() or 0
    next_id = max_id + 1
    return render_template('sensor_form.html', sensor=None, next_id=next_id, action='add')


@app.route('/db/sensor/edit/<int:sid>', methods=['GET', 'POST'])
def sensor_edit(sid):
    sensor = Sensor.query.get_or_404(sid)
    if request.method == 'POST':
        try:
            sensor.name        = request.form['name'].strip()
            sensor.unit        = request.form.get('unit', '').strip()
            sensor.description = request.form.get('description', '').strip()
            db.session.commit()
            flash(f'✅ Датчик #{sid} «{sensor.name}» обновлён.', 'success')
            return redirect(url_for('db_manager'))
        except Exception as e:
            db.session.rollback()
            flash(f'❌ Ошибка: {e}', 'danger')
    return render_template('sensor_form.html', sensor=sensor, next_id=sid, action='edit')


@app.route('/db/sensor/delete/<int:sid>', methods=['POST'])
def sensor_delete(sid):
    sensor = Sensor.query.get_or_404(sid)
    name   = sensor.name
    count  = Telemetry.query.filter_by(sensor_id=sid).count()
    try:
        db.session.delete(sensor)   # cascade удалит телеметрию
        db.session.commit()
        flash(f'🗑️ Датчик «{name}» и {count} связанных записей удалены.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ Ошибка: {e}', 'danger')
    return redirect(url_for('db_manager'))


# ══ CRUD ТЕЛЕМЕТРИЯ ══

@app.route('/db/telemetry/add', methods=['GET', 'POST'])
def telemetry_add():
    sensors = Sensor.query.order_by(Sensor.sensor_id).all()
    if request.method == 'POST':
        try:
            ts  = datetime.strptime(request.form['timestamp'], '%Y-%m-%dT%H:%M')
            sid = int(request.form['sensor_id'])
            val = float(request.form['value'])
            if not Sensor.query.get(sid):
                flash(f'❌ Датчик с ID {sid} не существует.', 'danger')
                return redirect(url_for('telemetry_add'))
            db.session.add(Telemetry(timestamp=ts, sensor_id=sid, value=val))
            db.session.commit()
            flash(f'✅ Запись добавлена: датчик {sid}, значение {val}.', 'success')
            return redirect(url_for('db_manager'))
        except Exception as e:
            db.session.rollback()
            flash(f'❌ Ошибка: {e}', 'danger')
    return render_template('telemetry_form.html',
                           record=None,
                           sensors=sensors,
                           now=datetime.now().strftime('%Y-%m-%dT%H:%M'),
                           action='add')


@app.route('/db/telemetry/edit/<int:rid>', methods=['GET', 'POST'])
def telemetry_edit(rid):
    record  = Telemetry.query.get_or_404(rid)
    sensors = Sensor.query.order_by(Sensor.sensor_id).all()
    if request.method == 'POST':
        try:
            record.timestamp = datetime.strptime(request.form['timestamp'], '%Y-%m-%dT%H:%M')
            record.sensor_id = int(request.form['sensor_id'])
            record.value     = float(request.form['value'])
            db.session.commit()
            flash(f'✅ Запись #{rid} обновлена.', 'success')
            return redirect(url_for('db_manager'))
        except Exception as e:
            db.session.rollback()
            flash(f'❌ Ошибка: {e}', 'danger')
    return render_template('telemetry_form.html',
                           record=record,
                           sensors=sensors,
                           now=record.timestamp.strftime('%Y-%m-%dT%H:%M'),
                           action='edit')


@app.route('/db/telemetry/delete/<int:rid>', methods=['POST'])
def telemetry_delete(rid):
    record = Telemetry.query.get_or_404(rid)
    try:
        db.session.delete(record)
        db.session.commit()
        flash(f'🗑️ Запись #{rid} удалена.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'❌ Ошибка: {e}', 'danger')
    return redirect(request.referrer or url_for('db_manager'))


# ══ SQL КОНСОЛЬ ══

ALLOWED_STATEMENTS = ('select', 'pragma')

@app.route('/db/sql', methods=['GET', 'POST'])
def sql_console():
    result_cols = []
    result_rows = []
    query_text  = ''
    error       = None
    exec_time   = None

    if request.method == 'POST':
        query_text = request.form.get('sql', '').strip()
        stmt = query_text.lower().lstrip()

        # Разрешаем только SELECT и PRAGMA
        if not any(stmt.startswith(kw) for kw in ALLOWED_STATEMENTS):
            error = '⛔ Разрешены только SELECT и PRAGMA запросы (защита данных).'
        else:
            try:
                import time
                t0 = time.time()
                res = db.session.execute(db.text(query_text))
                exec_time = round((time.time() - t0) * 1000, 2)
                result_cols = list(res.keys())
                result_rows = [list(row) for row in res.fetchall()]
            except Exception as e:
                error = f'❌ Ошибка SQL: {e}'

    # Подсказки
    hints = [
        'SELECT * FROM telemetry LIMIT 20;',
        'SELECT * FROM sensor;',
        'SELECT sensor_id, COUNT(*) as cnt FROM telemetry GROUP BY sensor_id;',
        'SELECT sensor_id, AVG(value) as avg_val, MIN(value) as min_val, MAX(value) as max_val FROM telemetry GROUP BY sensor_id;',
        'SELECT t.*, s.name, s.unit FROM telemetry t JOIN sensor s ON t.sensor_id = s.sensor_id LIMIT 10;',
        'SELECT * FROM telemetry ORDER BY value DESC LIMIT 10;',
        'PRAGMA table_info(telemetry);',
        'PRAGMA table_info(sensor);',
    ]

    return render_template('sql_console.html',
                           query_text=query_text,
                           result_cols=result_cols,
                           result_rows=result_rows,
                           error=error,
                           exec_time=exec_time,
                           hints=hints,
                           row_count=len(result_rows))

if __name__ == '__main__':
    app.run(debug=True)