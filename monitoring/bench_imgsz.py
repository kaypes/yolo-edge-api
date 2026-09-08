"""
monitoring/bench_imgsz.py
Aula 7 - Atividade 2, entregável 3: tabela comparativa imgsz=640 vs imgsz=320.

Sobe monitoring/yolo_monitor.py com o --imgsz pedido, amostra as métricas já
expostas por ele (porta 8000) e pelo Node Exporter (porta 9100) em intervalos
regulares durante --duration segundos, e imprime as médias de tempo de
inferência, uso de CPU, uso de memória e temperatura -- os números que vão
para a tabela comparativa, medidos e não estimados.

Uso:
    python3 monitoring/bench_imgsz.py --imgsz 640 --duration 300
    python3 monitoring/bench_imgsz.py --imgsz 320 --duration 300
"""
import argparse
import re
import statistics
import subprocess
import sys
import time
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--duration", type=int, default=300, help="Duração total da amostragem (s)")
    p.add_argument("--interval", type=int, default=5, help="Intervalo entre amostras (s)")
    p.add_argument("--warmup", type=int, default=15, help="Espera antes de começar a amostrar (s)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--video", default="monitoring/videos/transito.mp4")
    p.add_argument("--model", default="models/yolov8n.pt")
    return p.parse_args()


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read().decode()


def extract(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else None


def sample(yolo_url: str, node_url: str):
    yolo_text = fetch(yolo_url)
    node_text = fetch(node_url)

    inference_s = extract(yolo_text, r'^yolo_inference_time_seconds\s+([\d.eE+-]+)')

    mem_total = extract(node_text, r'^node_memory_MemTotal_bytes\s+([\d.eE+-]+)')
    mem_avail = extract(node_text, r'^node_memory_MemAvailable_bytes\s+([\d.eE+-]+)')
    mem_pct = (mem_total - mem_avail) / mem_total * 100 if mem_total and mem_avail else None

    load1 = extract(node_text, r'^node_load1\s+([\d.eE+-]+)')

    temps = [float(v) for v in re.findall(
        r'^node_hwmon_temp_celsius\{[^}]*chip="thermal_thermal_zone0"[^}]*\}\s+([\d.eE+-]+)',
        node_text, re.MULTILINE)]
    temp = max(temps) if temps else None

    return inference_s, load1, mem_pct, temp


def main():
    args = parse_args()
    yolo_url = f"http://localhost:{args.port}/metrics"
    node_url = "http://localhost:9100/metrics"

    print(f"[bench] Subindo yolo_monitor.py com imgsz={args.imgsz} ...")
    proc = subprocess.Popen([
        sys.executable, "monitoring/yolo_monitor.py",
        "--imgsz", str(args.imgsz),
        "--port", str(args.port),
        "--video", args.video,
        "--model", args.model,
    ])

    try:
        print(f"[bench] Aquecendo {args.warmup}s ...")
        time.sleep(args.warmup)

        n_samples = args.duration // args.interval
        inference_samples, load_samples, mem_samples, temp_samples = [], [], [], []

        print(f"[bench] Amostrando por {args.duration}s (a cada {args.interval}s, ~{n_samples} amostras) ...")
        for i in range(n_samples):
            inf, load, mem, temp = sample(yolo_url, node_url)
            if inf is not None:
                inference_samples.append(inf)
            if load is not None:
                load_samples.append(load)
            if mem is not None:
                mem_samples.append(mem)
            if temp is not None:
                temp_samples.append(temp)
            print(f"  [{i+1:>3}/{n_samples}] inference={inf} load={load} mem%={mem} temp={temp}")
            time.sleep(args.interval)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def avg(xs):
        return round(statistics.mean(xs), 4) if xs else None

    print()
    print(f"=== Resultado imgsz={args.imgsz} ({len(inference_samples)} amostras de inferência) ===")
    print(f"Tempo médio de inferência (s): {avg(inference_samples)}")
    print(f"Carga média de CPU (node_load1): {avg(load_samples)}")
    print(f"Uso médio de memória (%): {avg(mem_samples)}")
    print(f"Temperatura média (°C): {avg(temp_samples)}")


if __name__ == "__main__":
    main()
