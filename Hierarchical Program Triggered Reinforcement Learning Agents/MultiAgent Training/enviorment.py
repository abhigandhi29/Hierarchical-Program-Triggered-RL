import glob
import os
import sys
import math
import time
import numpy as np
import cv2
import random
from collections import deque

try:
	sys.path.append(glob.glob('/home/abhishek/Downloads/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg')[0])
except IndexError:
	pass
import carla

MAX_LEN = 1000
SECONDS_PER_EPISODE = 60
LIMIT_RADAR = 500
ROUNDING_FACTOR = 3

class CarlaMultiVehicles(object):
	def __init__(self,num_vehicles):
		self.client = carla.Client('localhost',2000)
		self.client.set_timeout(5.0)
		self.num_vehicles = num_vehicles
		self.radars_data = []
		for i in range(num_vehicles):
			self.radars_data.append(deque(maxlen=MAX_LEN))
        
	def reset(self,Norender:bool,loc_x:list, loc_y: list):
		if(len(loc_x)!=self.num_vehicles or len(loc_y)!=self.num_vehicles):
			assert(False)
		self.collision_hist = []
		for i in range(self.num_vehicles):
			self.collision_hist.append([])

		self.world = self.client.get_world()
		self.map = self.world.get_map()
		if Norender:
			settings = self.world.get_settings()
			settings.no_rendering_mode = True
			self.world.apply_settings(settings)
		
		#contains all the vehicles in the world
		self.vehicles = []
		self.sensors = []
		self.cosensors = []
  
		for i in range(self.num_vehicles):
			self.vehicles.append(0)
			self.sensors.append(0)
			self.cosensors.append(0)
		self.vehicle_id_to_idx = {}
		self.blueprint_library = self.world.get_blueprint_library()
		self.bp = self.blueprint_library.filter("model3")[0]
		colsensor = self.blueprint_library.find("sensor.other.collision")
		radar = self.blueprint_library.find('sensor.other.radar')
		radar.set_attribute("range", f"100")#Radar Data Collectiom
		radar.set_attribute("horizontal_fov", f"35")
		radar.set_attribute("vertical_fov", f"25")

		for i in range(self.num_vehicles):
			init_pos = carla.Transform(carla.Location(x=loc_x[i],y=loc_y[i]),carla.Rotation(yaw=180))
			self.vehicles[i] = self.world.spawn_actor(self.bp, init_pos)
			self.vehicle_id_to_idx[self.vehicles[i].id] = i
			#Create location to spawn sensors
			transform = carla.Transform(carla.Location(x=2.5, z=0.7))
			#Create Collision Sensors
			
			self.cosensors[i] = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicles[i])
			self.cosensors[i].listen(lambda event: self.collision_data(event))
   
			#Create location to spawn sensors
			transform = carla.Transform(carla.Location(x=2.5, z=0.7))
			

			#We will initialise Radar Data
			self.resetRadarData(100, 35, 25, i)

			
			self.sensors[i] = self.world.spawn_actor(radar, transform, attach_to=self.vehicles[i])
			self.sensors[i].listen(lambda data: self.process_radar(data))
			
   
		self.episode_start = time.time()
  
  
	

	def resetRadarData(self, dist, hfov, vfov, idx):
		# [Altitude, Azimuth, Dist, Velocity]
		alt = 2*math.pi/vfov
		azi = 2*math.pi/hfov

		vel = 0;
		deque_list = []
		for _ in range(MAX_LEN//4):
			altitude = random.uniform(-alt,alt)
			deque_list.append(altitude)
			azimuth = random.uniform(-azi,azi)
			deque_list.append(azimuth)
			distance = random.uniform(10,dist)
			deque_list.append(distance)
			deque_list.append(vel)
		self.radar_data[idx].extend(deque_list)
  
	def collision_data(self, event):
		for actor_id in event:
			self.collision_hist[self.vehicle_id_to_idx[actor_id]].append(event)
  
	#Process Camera Image
	def process_radar(self, radar):
		# To plot the radar data into the simulator
		for actor_id in radar:
			idx = self.vehicle_id_to_idx[actor_id]
			self._Radar_callback_plot(radar)

			# To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
			# Parameters :: frombuffer(data_input, data_type, count, offset)
			# count : Number of items to read. -1 means all data in the buffer.
			# offset : Start reading the buffer from this offset (in bytes); default: 0.
			points = np.frombuffer(buffer = radar.raw_data, dtype='f4')
			points = np.reshape(points, (len(radar), 4))
			for i in range(len(radar)):
				self.radar_data[idx].append(points[i,0])
				self.radar_data[idx].append(points[i,1])
				self.radar_data[idx].append(points[i,2])
				self.radar_data[idx].append(points[i,3])


	# Taken from manual_control.py
	def _Radar_callback_plot(self, radar_data):
		current_rot = radar_data.transform.rotation
		velocity_range = 7.5 # m/s
		world = self.world
		debug = world.debug

		def clamp(min_v, max_v, value):
			return max(min_v, min(value, max_v))

		for detect in radar_data:
			azi = math.degrees(detect.azimuth)
			alt = math.degrees(detect.altitude)
			# The 0.25 adjusts a bit the distance so the dots can
			# be properly seen
			fw_vec = carla.Vector3D(x=detect.depth - 0.25)
			carla.Transform(
			    carla.Location(),
			    carla.Rotation(
			        pitch=current_rot.pitch + alt,
			        yaw=current_rot.yaw + azi,
			        roll=current_rot.roll)).transform(fw_vec)
			
			norm_velocity = detect.velocity / velocity_range # range [-1, 1]
			r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
			g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
			b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
			
			debug.draw_point(
		        radar_data.transform.location + fw_vec,
		        size=0.075,
		        life_time=0.06,
		        persistent_lines=False,
		        color=carla.Color(r, g, b))


	def step(self, action, i):
		#Apply Vehicle Action
		self.vehicles[i].apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))

	#Method to take action by the DQN Agent for straight drive
	def step_straight(self, action, p):
		done = False
		for i in range(self.num_vehicles):
			self.step(action,i)
		#Calculate vehicle speed
		kmh = self.get_speed()

		if p:
			print(f'collision_hist----{self.collision_hist}------kmh----{kmh}------light----{self.vehicle.is_at_traffic_light()}')
		
		reward = 0
		for i in range(self.num_vehicles):

			if len(self.collision_hist) != 0:
				done = True
				reward = reward - 200
				break
			elif kmh[i]<2:
				done = False
				reward += -1
			elif kmh[i]<40:
				done = False
				reward += 1
				reward += float(kmh/10)
			elif kmh[i]<80:
				done = False
				reward += 8 - float(kmh/10)
			else:
				done = False
				reward += -1

		# Build in function of Carla
			if self.vehicles[i].is_at_traffic_light() and kmh<25:
				done = True
				reward = reward+100
			elif self.vehicles[i].is_at_traffic_light() and kmh>25:
				done = True
				reward = reward-100

		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True

		data = np.array(self.radar_data)

		return data[:][-LIMIT_RADAR:], [round(kmh, ROUNDING_FACTOR)] , reward, done, None


	def destroy(self):
		"""
			destroy all the actors
			:param self
			:return None
		"""
		print('destroying actors')
		for actor in self.vehicles:
			actor.destroy()
		for actor in self.cosensors:
			actor.destroy()
		for actor in self.sensors:
			actor.destroy()
		print('done.')

	def get_speed(self, idx=-1):
		"""
			Compute speed of a vehicle in Kmh
			:param vehicle: the vehicle for which speed is calculated
			:return: speed as a float in Kmh
		"""
		if idx!=-1:
			vel = self.vehicles[idx].get_velocity()
			return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

		vel = []
		for i in range(len(self.vehicles)):
			vel.append(self.get_speed(i))
		return vel

