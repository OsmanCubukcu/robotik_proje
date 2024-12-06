import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models  # type: ignore
from collections import deque
import random
#python=3.8.0, tensorflow=2.8.0, numpy=1.21.0, gym=0.26.2, matplotlib=3.4.3
###############################################Pusher Environment##################################################
# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        """Tampona bir geçiş ekler."""
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Tampondan rastgele bir minibatch seçer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1),
        )

    def size(self):
        """Tampondaki geçiş sayısını döndürür."""
        return len(self.buffer)

# Pusher Ortamı
class PusherEnv:
    def __init__(self):
        self.pusher_pos = np.array([5.0, 5.0])  # Ajan başlangıç pozisyonu
        self.target_pos = np.array([7.0, 7.0])  # Hedef pozisyonu
        self.action_space = np.array([0.5,0.5])  # Eylem alanı
        self.state = np.concatenate([self.pusher_pos, self.target_pos])  # Başlangıç durumu

    def step(self, action):
        action = np.array(action).flatten()

        # Aksiyonları ölçekle ve sınırla
        scaled_action = np.clip(action * 0.5, -0.5, 0.5)
        self.pusher_pos += scaled_action
        self.pusher_pos = np.clip(self.pusher_pos, 0, 10)


        distance_to_target = np.linalg.norm(self.pusher_pos - self.target_pos)
        
        if distance_to_target > 5:
            reward = -10 - distance_to_target*distance_to_target
        elif 4.5 < distance_to_target <= 5:
            reward = -8 - distance_to_target*distance_to_target
        elif 3.5 < distance_to_target <= 4.5:
            reward = -6 - distance_to_target*distance_to_target 
        elif 3 < distance_to_target <= 3.5:
            reward = -4 - distance_to_target*distance_to_target 
        elif 2.5 < distance_to_target <= 3:
            reward = 0 - distance_to_target*distance_to_target 
        elif 2 < distance_to_target <= 2.5:
            reward = 4 - distance_to_target*distance_to_target 
        elif 1.5 < distance_to_target <= 2:
            reward = 9 - distance_to_target*distance_to_target
        elif 1 < distance_to_target <= 1.5:
            reward = 25 - distance_to_target*distance_to_target 
        else:
            reward = 36 - distance_to_target*distance_to_target  

        done = distance_to_target < 0.1
        if done:
            reward += 100
            print("nihai ödül") 

        next_state = np.concatenate([self.pusher_pos, self.target_pos])
        return next_state, reward, done, {}


    def reset(self):
        self.pusher_pos = np.array([5.0, 5.0])  # Ajan başlangıç pozisyonuna döner
        self.state = np.concatenate([self.pusher_pos, self.target_pos])  # Yeni durum
        return self.state

    def render(self, initial=False, final=False):
        plt.figure(figsize=(6, 6))
        if initial:
            plt.plot(5.0, 5.0, 'bo', label='Pusher (Initial)', markersize=12)
            plt.plot(7.0, 7.0, 'ro', label='Target (Initial)', markersize=12)
        elif final:
            plt.plot(self.pusher_pos[0], self.pusher_pos[1], 'go', label='Pusher (Final)', markersize=12)
            plt.plot(self.target_pos[0], self.target_pos[1], 'ro', label='Target (Final)', markersize=12)
        else:
            plt.plot(self.pusher_pos[0], self.pusher_pos[1], 'bo', label='Pusher', markersize=12)
            plt.plot(self.target_pos[0], self.target_pos[1], 'ro', label='Target', markersize=12)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc="best")
        plt.title('Pusher and Target')
        plt.grid(True)
        plt.show()

#######################################################################################################
# SAC Ajanı
class SAC:
    def __init__(self, state_dim, action_dim, buffer_size=100000, entropy_weight=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self.build_actor()
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.target_critic1 = self.build_critic()
        self.target_critic2 = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        self.gamma = 0.99
        self.tau = 0.005

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        
        # Entropi ağırlığı
        self.entropy_weight = entropy_weight

        # Gürültü ayarı (ilk 250 adım için daha fazla rastgelelik)
        self.noise_scale = 1.0  # Gürültü skoru
        self.noise_increment = 0.01  # Gürültü artış miktarı

    def build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh')  # Continuous action space
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim + self.action_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1) 
        ])
        return model

    def get_action(self, state, step):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state)
        if step < 250:
            noise = np.random.normal(0, self.noise_scale, size=self.action_dim)
            action += noise
            action = np.clip(action, -1.0, 1.0)  

        return action[0] 

    def train(self, batch_size=64, step=None):
        if self.replay_buffer.size() < batch_size:
            return None, None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with tf.GradientTape(persistent=True) as tape:
            q1_value = self.critic1(tf.concat([states, actions], axis=-1))
            q2_value = self.critic2(tf.concat([states, actions], axis=-1))

            target_actions = self.actor(next_states)
            target_q1 = self.target_critic1(tf.concat([next_states, target_actions], axis=-1))
            target_q2 = self.target_critic2(tf.concat([next_states, target_actions], axis=-1))
            target_q = rewards + self.gamma * (1 - dones) * tf.maximum(target_q1, target_q2)

            critic1_loss = tf.reduce_mean(tf.square(q1_value - target_q))
            critic2_loss = tf.reduce_mean(tf.square(q2_value - target_q))

            # Actor kaybı hesaplanırken sabit bir entropi terimi eklenir
            actor_loss = -tf.reduce_mean(self.critic1(tf.concat([states, self.actor(states)], axis=-1))) + self.entropy_weight * tf.reduce_mean(-tf.math.log(self.actor(states)))

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target(self.target_critic1, self.critic1)
        self.update_target(self.target_critic2, self.critic2)

        # İlk 250 eğitim adımında gürültü miktarını arttır
        if step < 250:
            self.noise_scale += self.noise_increment

        return critic1_loss, critic2_loss, actor_loss

    def update_target(self, target, source):
        for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
            target_var.assign(target_var * (1 - self.tau) + source_var * self.tau)

def Create_Sac(train_num):
    env = PusherEnv()   
    state_dim = len(env.state)
    action_dim = len(env.action_space)   
    agent = SAC(state_dim, action_dim)  
    state = env.reset() 
    env.render(initial=True)
    
    for step in range(train_num):
        action = agent.get_action(state, step)  # Burada step parametresi eklenmeli
        next_state, reward, done, _ = env.step(action) 
        agent.replay_buffer.add((state, action, reward, next_state, done))
        critic1_loss, critic2_loss, actor_loss = agent.train(batch_size=64, step=step)  # Burada da step parametresi eklenmeli
        state = next_state
        if step % 100 == 0:
            print(f"Step: {step}, Pusher Position: {env.pusher_pos}, Reward: {reward}")
        elif step == train_num - 1: 
            print(f"Step: {step}, Pusher Position: {env.pusher_pos}, Reward: {reward}")           
    env.render(final=True)
################################################DDPG Agent##############################################
# DDPG Ajanı
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, buffer_capacity=100000, batch_size=32, entropy_weight=0.3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)
        self.gamma = 0.99  # Discount factor
        self.tau = 0.03  # Soft update rate for target networks
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight  # Entropi kat sayısı 

    def build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh'),
            layers.Lambda(lambda x: x * self.action_bound)  # Scale action output
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim + self.action_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)  # Value prediction
        ])
        return model

    def get_action(self, state, noise=0.3, random_exploration_steps=250, step=0):
        state = np.expand_dims(state, axis=0)

        # İlk 250 adım için rastgele eylemler
        if step < random_exploration_steps:
            action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)
        else:
            action = self.actor(state).numpy()[0]
            # Gürültü ekleme
            action = np.clip(action + noise * np.random.randn(self.action_dim), -self.action_bound, self.action_bound)
        
        return action

    def train(self):
        if self.buffer.size() < self.batch_size:
            return  # Tampon yeterli sayıda geçiş içermiyorsa, eğitim yapılmaz.

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        with tf.GradientTape(persistent=True) as tape:
            # Critic loss
            target_actions = self.target_actor(next_states)
            target_qs = rewards + self.gamma * (1 - dones) * self.target_critic(
                tf.concat([next_states, target_actions], axis=-1)
            )
            target_qs = tf.stop_gradient(target_qs)
            current_qs = self.critic(tf.concat([states, actions], axis=-1))
            critic_loss = tf.reduce_mean(tf.square(current_qs - target_qs))

            # Actor loss
            predicted_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(tf.concat([states, predicted_actions], axis=-1)))

            # Entropi ekleyelim
            # Entropi kaybı: eylem dağılımının çeşitliliğini artırır.
            action_distribution = tf.nn.softmax(predicted_actions)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(action_distribution * tf.math.log(action_distribution + 1e-8), axis=-1))
            actor_loss += self.entropy_weight * entropy_loss

        # Uygulama
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update
        self.update_target(self.target_actor, self.actor)
        self.update_target(self.target_critic, self.critic)

        return critic_loss, actor_loss

    def update_target(self, target, source):
        for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
            target_var.assign(target_var * (1 - self.tau) + source_var * self.tau)

def Create_Ddpg(num_train):
    env = PusherEnv()
    state_dim = len(env.state)
    action_dim = len(env.action_space)
    action_bound = 1.0

    agent = DDPG(state_dim, action_dim, action_bound)

    state = env.reset()
    
    env.render(initial=True)

    for step in range(num_train):
        # İlk 250 adımda rastgele keşif
        action = agent.get_action(state, noise=0.3, random_exploration_steps=250, step=step)
        next_state, reward, done, _ = env.step(action)
        
        agent.buffer.add((state, action, reward, next_state, done))
        
        losses = agent.train()
        if losses:
            critic_loss, actor_loss = losses
        state = next_state
        if step % 100 == 0:
            print(f"Step: {step}, Reward: {reward:.2f}, Pusher Position: {env.pusher_pos}")
        elif step == num_train-1:
            print(f"Step: {step}, Reward: {reward:.2f}, Pusher Position: {env.pusher_pos}")

    env.render(final=True)

###################################################PPO Agent###########################################
# PPO Ajanı
class PPO:
    def __init__(self, state_dim, action_dim, action_bound, buffer_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.buffer = ReplayBuffer(buffer_size)
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.gamma = 0.99  # Discount factor
        self.clip_ratio = 0.2  # PPO clip ratio
        self.max_grad_norm = 0.5  # Max gradient norm for clipping

    def build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh'),
            layers.Lambda(lambda x: x * self.action_bound)  # Scale action output
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)  # Value prediction
        ])
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)  # shape: (1, state_dim)
        action = self.actor(state).numpy()[0]  # Take the first (and only) output
        return action

    def get_value(self, state):
        state = np.expand_dims(state, axis=0)  # shape: (1, state_dim)
        return self.critic(state).numpy()[0][0]

    def train(self, states, actions, rewards, next_states, dones, batch_size=64):
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            ratio = tf.exp(tf.reduce_sum(actions_pred * actions, axis=1))  # Aksiyon oranı
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            advantage = rewards - self.critic(states)  # Monte Carlo avantajı: Geriye dönük toplam ödül
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))

            value_pred = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(rewards - value_pred))

            total_loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        grads = [tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm) for grad in grads]
        self.actor_optimizer.apply_gradients(zip(grads[:len(self.actor.trainable_variables)], self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads[len(self.actor.trainable_variables):], self.critic.trainable_variables))
    
    def compute_discounted_rewards(self, rewards, dones):
        """Discounted rewards hesaplar."""
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            cumulative_reward = rewards[t] + self.gamma * cumulative_reward * (1 - dones[t])
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards

    def train_step(self, env, agent, num_train, max_steps=20):
        env.render(initial=True)  # Başlangıç konumunu göster

        for episode in range(num_train):
            state = env.reset()
            done = False
            rewards, states, actions, next_states, dones = [], [], [], [], []

            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)
                state = next_state

                if done:
                    break

            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            # Monte Carlo avantajını hesapla
            discounted_rewards = self.compute_discounted_rewards(rewards, dones)
            agent.train(states, actions, discounted_rewards, next_states, dones)

            # Her 100 bölümde bir konum ve ödülü yazdır
            if episode % 100 == 0:
                total_reward = sum(rewards)
                print(f"Episode: {episode}, Final Position: {env.pusher_pos}, Total Reward: {total_reward/20}")
            elif episode == num_train-1:
                total_reward = sum(rewards)
                print(f"Episode: {episode}, Final Position: {env.pusher_pos}, Total Reward: {total_reward/20}")

        env.render(final=True)  # Eğitim sonrası son konumu göster

def Create_Ppo(num_train):
    # Çevre ve ajan başlat
    env = PusherEnv()
    agent = PPO(state_dim=4, action_dim=2, action_bound=1)
    agent.train_step(env, agent,num_train)



###################################################Arayüz###############################################
print("*"*40+"Blackjack'e Hosgeldiniz"+"*"*40)
while True:
    num_train=0
    num_test=0
    print("Lutfen islem secin=>")
    print("[1]Sac egitim=>")
    print("[2]DDPG egitim=>")
    print("[3]PPO egitim=>")
    print("[4]Quit=>")
    islem=""
    islem=(input())
    if(islem=="1"):   
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
        except ValueError:
            print("Hatali girdi!!!!!!")
        Create_Sac(num_train)
    elif(islem=="2"):
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
        except ValueError:
            print("Hatali girdi!!!!!!")
        Create_Ddpg(num_train)
    elif(islem=="3"):
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
        except ValueError:
            print("Hatali girdi!!!!!!")
        Create_Ppo(num_train)
    elif(islem=="4"):
        break
    else: 
        print("Hatali islem seçildi!!!!")
print("*"*103)