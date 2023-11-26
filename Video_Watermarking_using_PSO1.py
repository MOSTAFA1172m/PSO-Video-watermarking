import numpy as np
import cv2

# Define your update_velocity, update_position, and calculate_fitness functions here

def update_velocity(velocity, cognitive_component, social_component, global_best_position, inertia_weight):
    # Implement the update logic for particle velocity here 
    inertia_term = inertia_weight * velocity
    cognitive_term = cognitive_component
    social_term = social_component
    
    new_velocity = inertia_term + cognitive_term + social_term
    return new_velocity

def update_position(position, velocity, lower_bound, upper_bound):
    # Implement the update logic for particle position here
    new_position = position + velocity
    new_position = np.clip(new_position, lower_bound, upper_bound)  # this ensures that the positions stay within bounds
    return new_position

def calculate_fitness(encoded_frame, original_frame):
    # Implement the fitness function calculation here (MSE in this case) this is the objective function
    mse = np.mean((encoded_frame - original_frame) ** 2)
    return mse

def pso_algorithm(video_path, watermark_path, save_path, particle_size, convergence_threshold, lower_bound, upper_bound):
    # Load video and watermark images
    video = cv2.VideoCapture(video_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    watermark = watermark[:, :, :3]  # Keep only the first 3 channels (RGB)

    # Initialize particles with random positions and velocities
    particles = []
    for _ in range(particle_size):
        position = np.random.uniform(lower_bound, upper_bound, watermark.shape)
        velocity = np.random.rand(*watermark.shape)

        particles.append({
            'position': position,
            'velocity': velocity,
            'best_position': np.copy(position),
            'best_fitness': float('inf')
        })

    # Initialize global best position and fitness
    global_best_position = np.copy(particles[0]['position'])
    global_best_fitness = float('inf')

    # Define the PSO parameters        i put whatever i want :)
    
    inertia_weight = 0.7
    cognitive_weight = 1.5
    social_weight = 1.5

    # Main PSO loop with convergence criterion
    iteration = 0
    while True:
        for particle in particles:
            # Calculate fitness for the current particle
            fitness = calculate_fitness(particle['position'], watermark)

            # Update the particle's best position and fitness
            if fitness < particle['best_fitness']:
                particle['best_position'] = np.copy(particle['position'])
                particle['best_fitness'] = fitness

            # Update global best position and fitness
            if fitness < global_best_fitness:
                global_best_position = np.copy(particle['position'])
                global_best_fitness = fitness

            # Calculate cognitive and social components
            cognitive_component = cognitive_weight * np.random.rand(*watermark.shape) * (particle['best_position'] - particle['position'])
            social_component = social_weight * np.random.rand(*watermark.shape) * (global_best_position - particle['position'])

            # Update the particle's velocity
            particle['velocity'] = update_velocity(particle['velocity'], cognitive_component, social_component, global_best_position, inertia_weight)

            # Update the particle's position
            particle['position'] = update_position(particle['position'], particle['velocity'], lower_bound, upper_bound)

        print(f"Iteration {iteration + 1}, Best Fitness: {global_best_fitness}")
        
        # Check for convergence based on the threshold
        if global_best_fitness <= convergence_threshold or iteration == 20:
            break

        iteration += 1

    # Apply the watermark to the video using the global best position
    output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(video.get(3)), int(video.get(4))))

    while True:
        ret, frame = video.read()  #ret tells you to do the instruction or the routine
        if not ret:
            break

        # Resize the watermark to match the dimensions of the current frame
        watermark_resized = cv2.resize(watermark, (frame.shape[1], frame.shape[0]))

        watermark_applied = cv2.addWeighted(frame, 1.0, watermark_resized, 0.7, 0)
        output_video.write(watermark_applied)

    video.release()
    output_video.release()

if __name__ == "__main__":
    video_path = r"C:\Users\mosta\Downloads\pexels-maxime-g-18562546 (2160p).mp4"
    watermark_path = r"C:\Users\mosta\Downloads\Picture4.png"
    save_path = r"C:\Users\mosta\Downloads\watermarked_video1.mp4"  # You can specify the output path here
    particle_size = 10
    convergence_threshold = 0.001  # Set your desired convergence threshold
    lower_bound = 0
    upper_bound = 255

    pso_algorithm(video_path, watermark_path, save_path, particle_size, convergence_threshold, lower_bound, upper_bound)
