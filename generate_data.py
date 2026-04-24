"""
generate_data.py
Generates all 4 datasets for the Field Service Dispatch system.
Technician positions are snapped to real petrol station coordinates in Kuantan.
Jobs are also assigned to real petrol station locations.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── Reference point ────────────────────────────────────────────────────────
CENTER_LAT = 3.8063138
CENTER_LON = 103.3263902

# ── Real petrol station coordinates in Kuantan ─────────────────────────────
PETROL_STATIONS = [
    {"name": "Unnamed Station 1",    "lat": 3.843602,  "lon": 103.3389947},
    {"name": "Unnamed Station 2",    "lat": 3.9038326, "lon": 103.2468837},
    {"name": "Shell Kuantan 1",      "lat": 3.8157123, "lon": 103.3330641},
    {"name": "Smart Kuantan",        "lat": 3.7968744, "lon": 103.168695 },
    {"name": "BHPetrol Kuantan",     "lat": 3.9441146, "lon": 103.2420357},
    {"name": "Shell Kuantan 2",      "lat": 3.8050617, "lon": 103.3028328},
    {"name": "Petron Kuantan 1",     "lat": 3.8295294, "lon": 103.3418005},
    {"name": "Petronas Kuantan 1",   "lat": 3.8219602, "lon": 103.3265225},
    {"name": "Petronas Kuantan 2",   "lat": 3.7362248, "lon": 103.3069903},
    {"name": "Petronas Jaya Gading", "lat": 3.7561354, "lon": 103.2036068},
    {"name": "Petron Batu 3",        "lat": 3.7840716, "lon": 103.2880551},
    {"name": "Petron Kuantan 2",     "lat": 3.9233963, "lon": 103.366629 },
]

NUM_STATIONS = len(PETROL_STATIONS)

# ── Haversine distance ─────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ── Small jitter around a station (±200m) so techs aren't all at same point ─
def jitter(lat, lon, meters=200):
    dlat = np.random.uniform(-meters/111000, meters/111000)
    dlon = np.random.uniform(-meters/111000, meters/111000)
    return round(lat + dlat, 6), round(lon + dlon, 6)

# ═══════════════════════════════════════════════════════════════════════════
# 1. TECHNICIAN DATASET — 50 technicians
# ═══════════════════════════════════════════════════════════════════════════
NUM_TECHS  = 50
job_types  = ["Mechanical", "Electrical", "IT", "Civil"]
shifts     = ["Morning", "Evening"]

technicians = []
for i in range(1, NUM_TECHS + 1):
    # Assign technician to a real petrol station (cycle through all stations)
    station = PETROL_STATIONS[i % NUM_STATIONS]
    lat, lon = jitter(station["lat"], station["lon"], meters=300)

    # Status: most available during working hours
    status_weights = [0.60, 0.25, 0.15]   # Available, Working, Off Shift
    status = random.choices(
        ["Available", "Working", "Off Shift"],
        weights=status_weights
    )[0]

    technicians.append({
        "technician_id":       f"TECH{i:03d}",
        "technician_shift":    random.choice(shifts),
        "skill":               round(random.uniform(1, 5), 1),
        "status":              status,
        "job_type":            random.choice(job_types),
        "experience":          random.randint(1, 20),
        "nearest_station":     station["name"],
        "lat":                 lat,
        "lon":                 lon,
    })

df_tech = pd.DataFrame(technicians)
df_tech.to_csv("technician_dataset.csv", index=False)
print(f"✅ technician_dataset.csv — {len(df_tech)} rows")
print(f"   Available : {len(df_tech[df_tech['status']=='Available'])}")
print(f"   Working   : {len(df_tech[df_tech['status']=='Working'])}")
print(f"   Off Shift : {len(df_tech[df_tech['status']=='Off Shift'])}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. WORKLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════
workloads = []
for tech in technicians:
    workloads.append({
        "tech_id":      tech["technician_id"],
        "no_of_jobs":   random.randint(0, 15),
        "service_time": round(random.uniform(30, 240), 1),
        "travel_time":  round(random.uniform(5, 60), 1),
    })

df_workload = pd.DataFrame(workloads)
df_workload.to_csv("workload_dataset.csv", index=False)
print(f"✅ workload_dataset.csv — {len(df_workload)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# 3. JOB DATASET — all jobs at real petrol stations
# ═══════════════════════════════════════════════════════════════════════════
problems = [
    "Fuel pump failure",
    "POS terminal not working",
    "CCTV system down",
    "Air compressor breakdown",
    "Generator fault",
    "Electrical wiring issue",
    "Canopy lighting failure",
    "Underground tank sensor error",
    "Fire suppression system fault",
    "Car wash machine breakdown",
]
job_statuses = ["Submitted", "Ongoing", "Completed"]

# Map problem to most likely skill type
problem_skill_map = {
    "Fuel pump failure":             "Mechanical",
    "POS terminal not working":      "IT",
    "CCTV system down":              "IT",
    "Air compressor breakdown":      "Mechanical",
    "Generator fault":               "Electrical",
    "Electrical wiring issue":       "Electrical",
    "Canopy lighting failure":       "Electrical",
    "Underground tank sensor error": "IT",
    "Fire suppression system fault": "Civil",
    "Car wash machine breakdown":    "Mechanical",
}

base_time = datetime(2025, 1, 1, 8, 0, 0)
jobs = []
for i in range(1, NUM_STATIONS * 5 + 1):  # ~5 jobs per station = 60 jobs
    # Each job belongs to a real petrol station
    station = PETROL_STATIONS[i % NUM_STATIONS]
    lat, lon = jitter(station["lat"], station["lon"], meters=50)
    ts      = base_time + timedelta(hours=random.randint(0, 720))
    problem = random.choice(problems)

    jobs.append({
        "job_id":        f"JOB{i:04d}",
        "station_name":  station["name"],
        "priority":      random.randint(1, 5),
        "problems":      problem,
        "required_skill": problem_skill_map[problem],
        "timestamp":     ts.strftime("%Y-%m-%d %H:%M:%S"),
        "lat":           lat,
        "lon":           lon,
        "duration_exp":  random.randint(30, 180),
        "status":        random.choice(job_statuses),
    })

df_job = pd.DataFrame(jobs)
df_job.to_csv("job_dataset.csv", index=False)
print(f"✅ job_dataset.csv — {len(df_job)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SUPERVISION DATASET
# ═══════════════════════════════════════════════════════════════════════════
supervision = []
completed_jobs = df_job[df_job["status"].isin(["Ongoing", "Completed"])].copy()

for _, job in completed_jobs.iterrows():
    # Match technician skill to job requirement
    skilled_techs = [
        t for t in technicians
        if t["job_type"] == job["required_skill"] or t["skill"] >= 4.0
    ]
    tech = random.choice(skilled_techs if skilled_techs else technicians)

    req_ts    = datetime.strptime(job["timestamp"], "%Y-%m-%d %H:%M:%S")
    dist_km   = haversine(tech["lat"], tech["lon"], job["lat"], job["lon"])
    travel    = round(dist_km / 40 * 60, 1)
    arrival   = req_ts + timedelta(minutes=travel)
    duration  = round(random.uniform(30, job["duration_exp"] * 1.2), 1)
    complete  = arrival + timedelta(minutes=duration)
    cust_wait = round((arrival - req_ts).total_seconds() / 60, 1)

    supervision.append({
        "tech_id":              tech["technician_id"],
        "job_id":               job["job_id"],
        "station_name":         job["station_name"],
        "req_timestamp":        job["timestamp"],
        "dist":                 round(dist_km, 3),
        "duration":             duration,
        "cust_wait_time":       cust_wait,
        "exp_time_remaining":   max(0, round(job["duration_exp"] - duration, 1)),
        "arrival_time_only":    arrival.strftime("%H:%M:%S"),
        "completion_time_only": complete.strftime("%H:%M:%S"),
    })

df_supervision = pd.DataFrame(supervision)
df_supervision.to_csv("supervision_dataset.csv", index=False)
print(f"✅ supervision_dataset.csv — {len(df_supervision)} rows")

print("\n✅ All datasets generated successfully.")
print(f"\nPetrol stations covered: {NUM_STATIONS}")
for s in PETROL_STATIONS:
    dist = haversine(CENTER_LAT, CENTER_LON, s["lat"], s["lon"])
    print(f"  {s['name']:<30} ({s['lat']}, {s['lon']})  {dist:.1f} km from centre")
