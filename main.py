import gym
import tensorflow as tf
import os
import numpy as np
from numpy import random
from gym.spaces.box import Box
from CriticNet import *
from ActorNet import *
from collections import deque
import random




env = gym.make('Pendulum-v0')
batch_size = 32 #n. di esperienze per trainare - batch
y = .99 #Discount factor
num_episodes = 8000 #Numero episodi Train
max_epLength = 1000 #Lunghezza massima episodio
REPLAY_MEMORY_SIZE = 10000 #dimensione replay_buffer

#dimensione spazio
state_dim = env.observation_space.shape[0]
#numero azioni - in questo caso una solamente
action_dim = env.action_space.shape[0]
#intervallo azione tra [-2,2]
action_bound = env.action_space.high


#Creo le due reti, Actor e Critic
Actor = Actor_Net(state_dim,action_dim,action_bound)
Critic = Critic_Net(state_dim,action_dim,Actor.get_num_trainable_vars())

#inizializzo il replay buffer
replay_memory = []



session = tf.Session()
#per salvare i pesi
saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(session, checkpoint.model_checkpoint_path)
  print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old network weights")



total_steps = 0
step_counts = []

option = input("tape 0 for train, 1 for test\n")
if(option == 0):
  # Train
  for episode in range(num_episodes):
  	#ogni volta che un episodio finisce, reinizializzo l'ambiente
    state = env.reset()
    steps = 0
    totalrew = 0
    #ciclo interno
    for step in range(max_epLength):
      #env.render()
      action = None
      #noise per esplorazione
      action_ = session.run(Actor.scaled_out, feed_dict={Actor.input: [state]}) + (1. / (1. + episode))
      action = action_[0]
      #agisco nel mondo di gym, ricevendo reward e nuovo stato..   
      obs, actual_reward, done, _ = env.step(action)
      #riempo il buffer
      replay_memory.append((state, action, actual_reward, obs, done))
      #se buffer pieno, svuoto l'esperienza piu vecchia
      if len(replay_memory) > REPLAY_MEMORY_SIZE:
        replay_memory.pop(0)
      #
      state = obs
      #se ho riempito abbastanza il buffer(32 stati), posso iniziare la backprop
      if len(replay_memory) >= batch_size:
        minibatch = random.sample(replay_memory, batch_size)
        next_states = [m[3] for m in minibatch]
        #dall'actor prendo l'azione da eseguire, nel bound dell'intervallo [-2,2]
        actor_scaled = session.run(Actor.scaled_out_target, feed_dict={Actor.input: next_states})
        #dal Critic, usando come input l'azione calcolata nell'actor, calcolo il Q-Values
        q_values = session.run(Critic.out_fc3_target  , feed_dict={Critic.input: next_states,Critic.actions: actor_scaled})

        #----------------
        target_q = np.zeros(batch_size)
        target_action_mask = np.zeros((batch_size, 1), dtype=float)
        for i in range(batch_size):
          _, action, old_reward, _, done_ = minibatch[i]
          target_q[i] = old_reward
          if not done_:
            target_q[i] += y * q_values[i]
          target_action_mask[i] = action
        states = [m[0] for m in minibatch]
        #----------------
        
        #traino il critic - NB uso gli elementi del replay buffer
        _ = session.run(Critic.train_op, feed_dict={Critic.input: states,Critic.actions:target_action_mask,Critic.target:np.reshape(target_q, (32, 1))})        
        #traino l'actor - NB uso gli elementi del replay buffer
        action_scaled = session.run(Actor.scaled_out, feed_dict={Actor.input: states})    
        grads = session.run(Critic.action_grads, feed_dict={Critic.input: states,Critic.actions: action_scaled})
        _ = session.run(Actor.optimize, feed_dict={Actor.input: states,Actor.action_gradient: grads[0]})
        
        #updato il valore delle reti Target, poco alla volta, per evitare salti improvviso e destabilizazzione delle reti
        session.run(Actor.update_target_network_params)
        session.run(Critic.update_target_network_params)

      #dopo fine episodio - ciclo interno
      total_steps += 1
      totalrew += actual_reward
      steps += 1

      if done:
      	break
    
    #Statistiche
    step_counts.append(steps) 
    mean_steps = np.mean(step_counts[-100:])
    print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}"
                                    .format(episode, total_steps, mean_steps))
    print("rew")
    print totalrew
    if episode % 200 == 0:
      saver.save(session, 'saved_networks/' + '-dqn')

else: 
  # Test
  state = env.reset()
  total_rew = 0
  for episode in range(num_episodes):
     for step in range(max_epLength):     
      env.render()
      action = None
      action_ = session.run(Actor.scaled_out, feed_dict={Actor.input: [state]})
      action = action_[0]   
      state, actual_reward, done, _ = env.step(action)
      total_rew += actual_reward
      if done:
        env.reset()
        print "Finish with Rew "
        print total_rew
        total_rew = 0
        break
  