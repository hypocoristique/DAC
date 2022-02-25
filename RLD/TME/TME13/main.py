import gym
import gridworld
from utils import *
from core import *
import numpy as np
from agents import DQN
import torch


if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_gridworld.yaml', "HER-plan2")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DQN(env,config)


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        goal, _ = env.sampleGoal()
        goal = agent.featureExtractor.getFeatures(goal)
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            #env.render()
            pass

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                #env.render()
                pass

            ob = new_ob
            action= agent.act(ob, goal)
            new_ob, _, _, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            done = (new_ob == goal).all()
            reward = 1.0 if done else -0.1
            j+=1
            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")
            agent.store(ob, action, new_ob, reward, done, j, i, goal)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            
            if i % config["freqTarget"] == 0:
                agent.update_target()

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break

    env.close()