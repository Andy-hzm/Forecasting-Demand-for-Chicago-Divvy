"""
Major recurring Chicago public events, one row per event-day (2020-2026).

Kept as a small generator rather than a 200-line literal so adding a year is a
one-line edit. Dates were verified against official event sites / press
coverage on 2026-10-25. Cancelled years are simply absent; `note` records
reductions, moves and street-closure context.

Events: chicago_marathon, lollapalooza, pride_parade, air_and_water_show,
taste_of_chicago, st_patricks_parade, july4_fireworks, shamrock_shuffle,
bike_the_drive, bud_billiken_parade, jazz_festival, blues_festival, riot_fest,
auto_show, nascar_street_race, suenos_festival.
"""
from datetime import date, timedelta
LOC = {
 "grant_park": (41.8756, -87.6244), "millennium_park": (41.8826, -87.6226), "lakeview": (41.9400, -87.6446),
 "north_ave_beach": (41.9115, -87.6265), "navy_pier": (41.8917, -87.6086), "bronzeville": (41.8116, -87.6174),
 "douglass_park": (41.8620, -87.7010), "mccormick_place": (41.8512, -87.6165), "lake_shore_drive": (41.8818, -87.6194),
 "downtown": (41.8780, -87.6206),
}
rows = []
def add(ev, start, loc, note="", end=None):
    s = date.fromisoformat(start); e = date.fromisoformat(end) if end else s
    d = s
    while d <= e:
        rows.append((ev, d.isoformat(), loc, note)); d += timedelta(days=1)

# 1 marathon
for d in ["2021-10-10","2022-10-09","2023-10-08","2024-10-13","2025-10-12","2026-10-11"]: add("chicago_marathon", d, "grant_park")
# 2 lollapalooza
for s,e in [("2021-07-29","2021-08-01"),("2022-07-28","2022-07-31"),("2023-08-03","2023-08-06"),("2024-08-01","2024-08-04"),("2025-07-31","2025-08-03"),("2026-07-30","2026-08-02")]: add("lollapalooza", s, "grant_park", end=e)
# 3 pride
add("pride_parade","2021-10-03","lakeview","2021 parade cancelled (covid); Northalsted Pride Fest extended to Oct 3 in its place")
for d in ["2022-06-26","2023-06-25","2024-06-30","2025-06-29","2026-06-28"]: add("pride_parade", d, "lakeview")
# 4 air and water
add("air_and_water_show","2021-08-21","north_ave_beach","reduced: Blue Angels-only show, noon-1pm", end="2021-08-22")
for s,e in [("2022-08-20","2022-08-21"),("2023-08-19","2023-08-20"),("2024-08-10","2024-08-11"),("2025-08-16","2025-08-17"),("2026-08-15","2026-08-16")]:
    add("air_and_water_show", s, "north_ave_beach", "moved a week early for DNC" if s.startswith("2024") else "", end=e)
# 5 taste
for s,e,n in [("2022-07-08","2022-07-10","scaled back, ticketless"),("2023-09-08","2023-09-10","moved to Sept for NASCAR"),("2024-09-06","2024-09-08","moved to Sept for NASCAR/DNC"),("2025-09-05","2025-09-07","moved to Sept for NASCAR"),("2026-07-08","2026-07-12","back to July, 5 days")]:
    add("taste_of_chicago", s, "grant_park", n, end=e)
# 6 st pats
for d in ["2022-03-12","2023-03-11","2024-03-16","2025-03-15","2026-03-14"]: add("st_patricks_parade", d, "downtown", "river dyeing same morning")
# 7 july 4 navy pier
add("july4_fireworks","2022-07-02","navy_pier","Navy Pier holiday fireworks held Sat Jul 2 9pm; no show on Jul 4")
add("july4_fireworks","2023-07-01","navy_pier","Navy Pier holiday fireworks held Sat Jul 1 9pm; no show on Jul 4 (regular show Jul 5)")
add("july4_fireworks","2024-07-03","navy_pier","Navy Pier/NASCAR/Choose Chicago expanded fireworks Wed Jul 3; no show on Jul 4 (regular show Jul 6)")
add("july4_fireworks","2025-07-05","navy_pier","Navy Pier 15-min holiday show Sat Jul 5 10pm; no show on Jul 4 (regular show Jul 2)")
add("july4_fireworks","2026-07-04","navy_pier","Navy Pier largest-ever 15-min show on Jul 4 (US 250th)")
# 8 shamrock shuffle
for d in ["2022-03-20","2023-03-26","2024-03-24","2025-03-23","2026-03-22"]: add("shamrock_shuffle", d, "grant_park", "2020 cancelled, 2021 virtual" if d.startswith("2022") else "")
# 9 bike the drive
for d in ["2021-09-05","2022-09-04","2023-09-03","2024-09-01","2025-08-31","2026-09-06"]: add("bike_the_drive", d, "lake_shore_drive", "Labor Day Sunday; LSD closed ~6am-11am")
# 10 bud billiken
add("bud_billiken_parade","2021-08-14","bronzeville","scaled-down edition, half participants")
for d in ["2022-08-13","2023-08-12","2024-08-10","2025-08-09","2026-08-08"]: add("bud_billiken_parade", d, "bronzeville")
# 11 jazz
add("jazz_festival","2021-09-04","millennium_park","no full festival in 2021; single Chicago In Tune jazz night at Pritzker Pavilion")
for s,e in [("2022-09-01","2022-09-04"),("2023-08-31","2023-09-03"),("2024-08-29","2024-09-01"),("2025-08-28","2025-08-31"),("2026-09-03","2026-09-06")]:
    add("jazz_festival", s, "millennium_park", "Thu opener typically at Cultural Center/Pritzker", end=e)
# 12 blues
add("blues_festival","2021-09-18","millennium_park","no June festival in 2021; single Chicago In Tune blues night at Pritzker Pavilion")
for s,e in [("2022-06-09","2022-06-12"),("2023-06-08","2023-06-11"),("2024-06-06","2024-06-09"),("2025-06-05","2025-06-08"),("2026-06-04","2026-06-07")]:
    add("blues_festival", s, "millennium_park", end=e)
# 13 riot fest
add("riot_fest","2021-09-17","douglass_park","preview party Sep 16", end="2021-09-19")
for s,e in [("2022-09-16","2022-09-18"),("2023-09-15","2023-09-17"),("2024-09-20","2024-09-22"),("2025-09-19","2025-09-21"),("2026-09-18","2026-09-20")]:
    add("riot_fest", s, "douglass_park", end=e)
# 14 auto show
for s,e,n in [("2020-02-08","2020-02-17",""),("2021-07-15","2021-07-19","special July edition"),("2022-02-12","2022-02-21",""),("2023-02-11","2023-02-20",""),("2024-02-10","2024-02-19",""),("2025-02-08","2025-02-17",""),("2026-02-07","2026-02-16","")]:
    add("auto_show", s, "mccormick_place", n, end=e)
# 15 nascar
add("nascar_street_race","2023-07-01","grant_park","parking restrictions from Jun 2, Ida B. Wells closed Jun 10, Balbo Jun 19, major closures from Jun 25; teardown into mid-July", end="2023-07-02")
add("nascar_street_race","2024-07-06","grant_park","street closures began Jun 10 (Ida B. Wells); 38-day footprint", end="2024-07-07")
add("nascar_street_race","2025-07-05","grant_park","street closures began Jun 19 (Balbo); 25-day footprint; race paused for 2026", end="2025-07-06")
# 16 suenos
for s,e in [("2022-05-28","2022-05-29"),("2023-05-27","2023-05-28"),("2024-05-25","2024-05-26"),("2025-05-24","2025-05-25"),("2026-05-23","2026-05-24")]:
    add("suenos_festival", s, "grant_park", "Memorial Day weekend", end=e)

rows.sort(key=lambda r: (r[1], r[0]))

CHICAGO_EVENTS = [
    {"event": ev, "date": d, "location": loc, "lat": LOC[loc][0], "lng": LOC[loc][1], "note": note}
    for ev, d, loc, note in rows
]
