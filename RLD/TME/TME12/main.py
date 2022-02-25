import argparse

import numpy as np

from BehavioralCloning import BehavioralCloning
from core import *
from GAIL import GailAgent
from utils import checkConfUpdate, init

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--mode", default=0, type=int, help="Mode 0: Behavioral Cloning; Mode 1: GAIL")

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode ==0:
        env, config, outdir, logger = init('./configs/config_cloning_lunar.yaml', "BehavorialCloning")
        agent = BehavioralCloning(env,config, logger)
    if args.mode == 1:
        env, config, outdir, logger = init('./configs/config_gail_lunar.yaml', "GAIL")
        agent = GailAgent(env,config, logger)
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    track_reward = []
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
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
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j, i)
            rsum += reward

            if agent.timeToLearn(done):
                log_dict = agent.learn()
            
            # if i % config["freqTarget"] == 0:
            #     agent.update_target()

            if done:
                track_reward.append(rsum)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                #logger.direct_write("loss", log_dict["loss"], i)
                logger.direct_write("trailing-reward", sum(track_reward[-100:]) / len(track_reward[-100:]), i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
