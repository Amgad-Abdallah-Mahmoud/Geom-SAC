from MolGraphEnv import *
from agent import *
from utils import get_final_mols

if __name__ == "__main__":
    scores = []
    mols = []
    top = []
    actor_loss = []
    n_episodes = 5000
    ref_mol = Chem.MolFromSmiles('CC(=CCC1=CC2=C(C(=C1OC)O)C(=O)C3=C(O2)C=CC(=C3)O)C')

    for episode in range(n_episodes):
        rn = np.random.choice(["O=C1CSC(=Nc2cc(F)ccc2Cc2cncs2)N1", ])
        init_mol = Chem.MolFromSmiles(rn)
        env = MolecularGraphEnv(mol_g=init_mol, reference_mol=ref_mol, target_sim=1, max_atom=40)
        state = env.reset(frame_work='pyg')
        graph_encoder = GraphEncoder(state)
        state = graph_encoder(state)
        agent = SoftActorCriticAgent(env, state)
        rewards = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            probabilities, actions, log_p = agent.select_actions(state)
            next_state, reward, done, info = env.step(actions[0].detach().cpu().numpy())
            graph_encoder_ = GraphEncoder(next_state)
            next_state = graph_encoder_(next_state)
            agent.replay_buffer.add(
                (state.detach().view(-1), probabilities.view(-1), reward, next_state.detach().view(-1), done,))
            agent.train()
            state = next_state
            rewards += reward

            if done:
                break

        scores.append(rewards)
        print("steps: ", steps)
        actor_loss.append([x.item() for x in agent.ac_loss])
        try:
            mols.append(Chem.MolToSmiles(env.get_final_mol()))
        except:
            pass
        try:
            if Chem.QED.qed(env.get_final_mol()) > 0.79:
                top.append(Chem.MolToSmiles(env.get_final_mol()))
            if Chem.QED.qed(get_final_mols(env.get_final_mol())) > 0.79:
                top.append(Chem.MolToSmiles(env.get_final_mol()))
        except:
            pass
