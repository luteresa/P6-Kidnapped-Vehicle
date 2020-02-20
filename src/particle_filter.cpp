/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

#include <limits>


using std::string;
using std::vector;
using std::normal_distribution;
std::default_random_engine gen;

#define EPS 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if (is_initialized) {
    return;
  }
  std::default_random_engine gen;
  double std_x, std_y, std_theta;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  num_particles = 60;  // TODO: Set the number of particles
  for (int i=0; i < num_particles; i++) {
		Particle p;
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1;

		particles.push_back(p);
  }

  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];
  normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);
	for (unsigned int i=0; i < particles.size(); i++) {
		double x_ = particles[i].x;
		double y_ = particles[i].y;
		double theta_ = particles[i].theta;
		if (fabs(yaw_rate) >= EPS) {
      x_ = x_ + (velocity/yaw_rate)*(sin(theta_+yaw_rate*delta_t)-sin(theta_));
      y_ = y_ + (velocity/yaw_rate)*(cos(theta_)-cos(theta_+yaw_rate*delta_t));
      theta_ = theta_ + yaw_rate*delta_t;
    } else {
      x_ = x_ + velocity * delta_t*cos(theta_);
      y_ = y_ + velocity * delta_t*sin(theta_);
    }
		
		particles[i].x = x_ + dist_x(gen);
		particles[i].y = y_ + dist_y(gen);
		particles[i].theta = theta_ + dist_theta(gen);
	}
}

//匹配最近邻地标点
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  int num_obs= observations.size();
	int num_pred = predicted.size();

	for (int i = 0; i < num_obs; i++) {
		//for each observation
		double min_disance = std::numeric_limits<double>::max();

		//initializing the found map that is not in map , this is made for return the nearset measurement around GT.
		int id_in_map = -1;
		//complexity is o(ij);
		for (int j = 0; j < num_pred; j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < min_disance) {
				min_disance = distance;
				id_in_map = predicted[j].id;
			}
		}
		observations[i].id = id_in_map;
	}

}
double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

	double sig_x, sig_y;
	sig_x = std_landmark[0];
	sig_y = std_landmark[1];


  // 循环处理每个粒子
	for (unsigned int i=0; i < particles.size(); i++) {
		double x_part = particles[i].x;
		double y_part = particles[i].y;
		double theta_ = particles[i].theta;


		//step0:去掉超出物理测试范围的landmark
		double sensor_range_2 = sensor_range * sensor_range;
		vector<LandmarkObs> inRangeLandmarks;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmarkX = map_landmarks.landmark_list[j].x_f;
			float landmarkY = map_landmarks.landmark_list[j].y_f;
			int id = map_landmarks.landmark_list[j].id_i;
			double dX = x_part - landmarkX;
			double dY = y_part - landmarkY;

			//in this step, in range is constructed. After this step, we only calculate the landmarks in the range. 
			if (dX*dX + dY * dY <= sensor_range_2) {
				inRangeLandmarks.push_back(LandmarkObs{ id, landmarkX, landmarkY });
			}
		}

		// step1:对所有检测点坐标系转换，齐次变换
		// Transfrom observation coodinates from vehicle coordinate to map (global) coordinate.
		vector<LandmarkObs> mappedObservations;
		//Rotation
		for (unsigned int j = 0; j< observations.size(); j++) {
			double x_map,y_map;
      //此处观测点，一个传感器对应一个序列值，对多个不同粒子，只是x_part,y_part不同
      //因此，对多个粒子，只需要保持一个备份即可，在粒子太多时，作此优化可以节省时间
			x_map = x_part + (cos(theta_) * observations[j].x) - (sin(theta_) * observations[j].y);
			y_map = y_part + (sin(theta_) * observations[j].x) + (cos(theta_) * observations[j].y);
			mappedObservations.push_back(LandmarkObs{ observations[j].id, x_map, y_map });
		}

		//step2:找到匹配标记(最邻近法)
		dataAssociation(inRangeLandmarks, mappedObservations);

		double final_weight=1.0;
		
		//step3: calculate the weights
		for (unsigned int j = 0; j < mappedObservations.size(); j++) {
			double observation_x = mappedObservations[j].x;
			double observation_y = mappedObservations[j].y;
			int landmark_id = mappedObservations[j].id;

			double landmark_x, landmark_y;
			for (unsigned int k=0; k < inRangeLandmarks.size();k++) {
				if (landmark_id == inRangeLandmarks[k].id) {
					landmark_x = inRangeLandmarks[k].x;
					landmark_y = inRangeLandmarks[k].y;
					break;
				}
			}
			final_weight *= multiv_prob(sig_x, sig_y, observation_x, observation_y, landmark_x, landmark_y);
		}
		particles[i].weight = final_weight;
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
 // Get weights and max weight.
  vector<double> weights;
  double max_weight = std::numeric_limits<double>::min();
  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if ( particles[i].weight > max_weight ) {
      max_weight = particles[i].weight;
    }
  }

  // Creating distributions.
  std::uniform_real_distribution<float> dist_float(0.0, max_weight);
  std::uniform_int_distribution<int> dist_int(0, num_particles - 1);

  // Generating index.
  int index = dist_int(gen);

  double beta = 0.0;

  // the wheel
  vector<Particle> resampled_particles;
  for(int i = 0; i < num_particles; i++) {
    beta += dist_float(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}