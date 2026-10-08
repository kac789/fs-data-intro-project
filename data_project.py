import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd

#define constants
gear_ratio = 12/41
pi = np.pi
wheel_radius = 0.2
m_eff = 244.08
m = 221.4
g = 9.81
p = 1.225


path = "/Users/kylerchien/fs-5-software-intro-projects/data/software-data.parquet"
#1 read parquet 
frame = pl.read_parquet(path)

def speed_calculation(RPM):
    '''
    Takes in RPM and converts into speed 
    '''
    v = RPM * ((2*pi)/60) * wheel_radius * gear_ratio
    return v
#print(frame.columns)
print(frame)

#2 speed charting 
speed = speed_calculation(frame["SME_TRQSPD_Speed"]) 
time = frame["Time"]

plt.plot(time, speed, color = 'blue', linestyle = "solid")
plt.title("Cars Speed")
plt.xlabel("Time")
plt.ylabel("Speed")

#3 acceleration, braking, coasting 
braking = frame["ETC_STATUS_BRAKE_SENSE_VOLTAGE"].fill_null(strategy = "forward")

#remove the noise from code
smoothed_speed = speed.rolling_mean(window_size = 42)
acceleration = np.gradient(smoothed_speed,time)
df_acceleration = pl.DataFrame({
    "acceleration": acceleration,
    "time": time,
    "speed": speed,
    "brake": braking,
})
print(df_acceleration.head())


#plot acceleration
plt.figure()
plt.plot(time, acceleration, color = 'blue', linestyle = "solid")
plt.title("Acceleration")


#plot acceleration with window based upon low and high 
filter_acceleration = df_acceleration.filter(pl.col("acceleration") > 0.5)
new_time = filter_acceleration["time"]
new_acceleration = filter_acceleration["acceleration"]

#clean some values 
filter_time = df_acceleration.filter(pl.col("time") > 55).filter(pl.col("time") <= 65)
print(filter_time["acceleration"].min())
plt.figure()
plt.plot(new_time, new_acceleration, color = 'blue', linestyle = "solid")
plt.title("Acceleration Window")

#plot coasting
plt.figure()
#speed greater than 1 means moving somewhat 
filter_coasting = df_acceleration.filter(pl.col("acceleration") > -0.77).filter(pl.col("acceleration") <= 0.12).filter(pl.col("speed") > 1).filter(pl.col("brake") < 375)
coasting_speed = filter_coasting["speed"]
coasting_time = filter_coasting["time"]
plt.plot(coasting_time, coasting_speed, color = 'blue', linestyle = "none", marker = ".")
plt.title("Coasting Window")
plt.xlabel("Time")
plt.ylabel("Accleration")

#plot braking 
plt.figure()
plt.plot(time, braking, color = 'blue', linestyle = "solid")
plt.title("Braking")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.axhline(y=375, color = "red", linestyle = "--", label = "threshold")

#4 laps and lap speed


gps_speed = frame["VDM_GPS_SPEED"]
gps_latitude = frame["VDM_GPS_Latitude"]
gps_longitude = frame["VDM_GPS_Longitude"]
print(gps_speed.max())

df_gps = pl.DataFrame({
    "time": time,
    "latitude": gps_latitude,
    "longitude": gps_longitude,
})
#scatter chart
clean_df_gps = df_gps.drop_nulls()
plt.figure()
plt.scatter(gps_longitude, gps_latitude, c = df_gps["time"])
plt.title("Lap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar()

#scatter plot 
plt.figure()
x = clean_df_gps["time"]
y = clean_df_gps["longitude"]
plt.plot(x, y, color = 'blue', linestyle = "solid")

#5 max speed and acceleration
print(df_acceleration["speed"].max())
print(df_acceleration["acceleration"].max())
#acceleration window 
acceleration_window1 = filter_acceleration.filter(pl.col("time") > 0).filter(pl.col("time") <= 100)
print(acceleration_window1["speed"].max())
print("acceleration1: ", acceleration_window1["acceleration"].max())
print(acceleration_window1.height*0.0119)
acceleration_window2 = filter_acceleration.filter(pl.col("time") > 100).filter(pl.col("time") <= 160)
print("acceleration2: ", acceleration_window2["acceleration"].max())
print(acceleration_window2["speed"].max())
print(acceleration_window2.height*0.0119)
coasting_window1 = filter_coasting.filter(pl.col("time") > 0).filter(pl.col("time") <= 100)
print(coasting_window1.height*0.0119)
coasting_window2 = filter_coasting.filter(pl.col("time") > 100).filter(pl.col("time") <= 160)
print(coasting_window2.height * 0.0119)


#6 filter coasting data
clean_coasting = filter_coasting.filter(pl.col("speed") > 5)
coasting_gaps = clean_coasting.with_columns(
    gap = pl.col("time").diff()
)
coasting_chunks = coasting_gaps.with_columns(
    chunk = (pl.col("gap") > 0.1).fill_null(False).cum_sum()
)
chunk_length = coasting_chunks.group_by("chunk").agg(
    length = pl.col("time").max() - pl.col("time").min()
)
print(chunk_length.sort("chunk"))
plt.figure()
long_chunks = chunk_length.filter(pl.col("length") >= 1)["chunk"].to_list()
final_coasting = coasting_chunks.filter(pl.col("chunk").is_in(long_chunks))
a = final_coasting["time"]
b = final_coasting["speed"]
plt.plot(a, b, color = 'blue', linestyle = "solid")

#7 physics on it 
x = final_coasting["speed"] ** 2
y = final_coasting["acceleration"]
plt.figure()
plt.scatter(x,y)
slope, intercept = np.polyfit(x, y, 1)
y_line = (slope * x) + intercept
plt.plot(x, y_line, color = 'red')
drag_term = -slope * m_eff
print(drag_term)
cdA = drag_term/(0.5 *p)
crr = (-intercept * m_eff) / (m*g)

print(cdA)
print(crr)
plt.show()



