import numpy as np
from deep_lagrangian_networks.utils import load_dataset
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib as mp
import time

n_dof = 2

with open('data/character_data.pickle', 'rb') as f:
    temp = pickle.load(f)

# train, test
qp_train, qp_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
qv_train, qv_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
qa_train, qa_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
tau_train, tau_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
m_train, m_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
c_train, c_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
g_train, g_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
p_train, p_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))
pd_train, pd_test = np.zeros((0, n_dof)), np.zeros((0, n_dof))

test_label = ("e", "q", "v")
test_idx = [temp["labels"].index(x) for x in test_label]

for i in range(len(temp["labels"])):
    if i in test_idx:
        qp_test = np.vstack((qp_test, temp["qp"][i]))
        qv_test = np.vstack((qv_test, temp["qv"][i]))
        qa_test = np.vstack((qa_test, temp["qa"][i]))
        pd_test = np.vstack((pd_test, temp["pdot"][i]))
        p_test = np.vstack((p_test, temp["p"][i]))

        tau_test = np.vstack((tau_test, temp["tau"][i]))
        m_test = np.vstack((m_test, temp["m"][i]))
        c_test = np.vstack((c_test, temp["c"][i]))
        g_test = np.vstack((g_test, temp["g"][i]))
    else:
        qp_train = np.vstack((qp_train, temp["qp"][i]))
        qv_train = np.vstack((qv_train, temp["qv"][i]))
        qa_train = np.vstack((qa_train, temp["qa"][i]))
        pd_train = np.vstack((pd_train, temp["pdot"][i]))
        p_train = np.vstack((p_train, temp["p"][i]))

        tau_train = np.vstack((tau_train, temp["tau"][i]))
        m_train = np.vstack((m_train, temp["m"][i]))
        c_train = np.vstack((c_train, temp["c"][i]))
        g_train = np.vstack((g_train, temp["g"][i]))

print("\n\n################################################")
print("Characters:")
print("   Test Characters = {}".format(len(test_label)))
print("  Train Characters = {}".format(len(temp["labels"]) - len(test_label)))
print("# Training Samples = {0:05d}".format(int(qp_train.shape[0])))
print("")

##########################################################################################################################################

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

seq = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 8)
)

epoch_limit = 10000
criterion = torch.nn.MSELoss()  # binary cross entropy
optimizer = torch.optim.Adam(seq.parameters(), lr=5.e-04, weight_decay=1.e-5, amsgrad=True)
st = time.time()
total_loss = 0
for epoch in range(epoch_limit):
    running_loss = 0.0
    if epoch % 10 == 0:
        epoch10_st = time.time()
    epoch_st = time.time()
    for i in range(len(qp_train)):
        #train
        seq.train()
        optimizer.zero_grad()
        tensor_in = torch.tensor(
            [qp_train[i, 0], qv_train[i, 0], qa_train[i, 0], pd_train[i, 0], p_train[i, 0], qp_train[i, 1],
             qv_train[i, 1], qa_train[i, 1], pd_train[i, 1], p_train[i, 1]], dtype=torch.float32)
        tensor_out = torch.tensor(
            [tau_train[i, 0], m_train[i, 0], c_train[i, 0], g_train[i, 0], tau_train[i, 1], m_train[i, 1],
             c_train[i, 1], g_train[i, 1]], dtype=torch.float32)
        out = seq(tensor_in)
        loss = criterion(out, tensor_out)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_et = time.time()
    total_loss += running_loss
    if epoch % 10 == 9:
        epoch10_et = time.time()
        print(
            f'{((epoch + 1) / epoch_limit):.2%} \nEpoch {epoch + 1} - Time for 10 epochs: {(epoch10_et - epoch10_st):.3f} s \nLoss for last: {running_loss}')
        print("\n################################################")
    #print(f'Epoch {epoch} - Time: {(epoch_et-epoch_st):.3f} s Loss: {running_loss}')
et = time.time()
total_time = et - st
seq.eval()
print("/////////////////////////////////////////////////////////////////")
print(f"Time training for {epoch_limit}: {total_time:.3f} s")
print(f"avg loss: {total_loss / epoch_limit}")
print("/////////////////////////////////////////////////////////////////")
PATH = f'nn{epoch_limit}.pth'
torch.save(seq.state_dict(), PATH)
nn_tau = np.empty((len(tau_test), 2))
nn_m = np.empty((len(tau_test), 2))
nn_c = np.empty((len(tau_test), 2))
nn_g = np.empty((len(tau_test), 2))
t0_evaluation = time.perf_counter()
for j in range(len(tau_test)):
    tensor_test = torch.tensor(
        [qp_test[j, 0], qv_test[j, 0], qa_test[j, 0], pd_test[j, 0], p_test[j, 0], qp_test[j, 1], qv_test[j, 1],
         qa_test[j, 1], pd_test[j, 1], p_test[j, 1]], dtype=torch.float32)
    test = np.array(tensor_test.tolist())
    result=seq(tensor_test)
    for i in range(4):
        if i % 4 == 0:
            nn_tau[j, 0] = result[i % 4]
            nn_tau[j, 1] = result[(i % 4) + 4]
        elif i % 4 == 1:
            nn_m[j, 0] = result[i % 4]
            nn_m[j, 1] = result[(i % 4) + 4]
        elif i % 4 == 2:
            nn_c[j, 0] = result[i % 4]
            nn_c[j, 1] = result[(i % 4) + 4]
        elif i % 4 == 3:
            nn_g[j, 0] = result[i % 4]
            nn_g[j, 1] = result[(i % 4) + 4]
t_eval = (time.perf_counter() - t0_evaluation) / float(qp_test.shape[0])

# Compute Errors:
err_g = 1. / float(qp_test.shape[0]) * np.sum((nn_g - g_test) ** 2)
err_m = 1. / float(qp_test.shape[0]) * np.sum((nn_m - m_test) ** 2)
err_c = 1. / float(qp_test.shape[0]) * np.sum((nn_c - c_test) ** 2)
err_tau = 1. / float(qp_test.shape[0]) * np.sum((nn_tau - tau_test) ** 2)

print("\nPerformance:")
print("                Torque MSE = {0:.3e}".format(err_tau))
print("              Inertial MSE = {0:.3e}".format(err_m))
print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
print("         Gravitational MSE = {0:.3e}".format(err_g))
print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))
print("\n################################################")

with open(f'nn_{epoch_limit}.txt', 'w') as file:
    info = [
        "Performance:",
        "                Torque MSE = {0:.3e}".format(err_tau),
        "              Inertial MSE = {0:.3e}".format(err_m),
        "Coriolis & Centrifugal MSE = {0:.3e}".format(err_c),
        "         Gravitational MSE = {0:.3e}".format(err_g),
        "      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1. / t_eval),
        "                Train time = {0:.3e}s".format(total_time)
    ]
    for i in info:
        file.write(i + '\n')
#################################################################################################################################################################################

_, _, divider, _ = load_dataset(filename='data/character_data.pickle')
ticks = np.array(divider)
ticks = (ticks[:-1] + ticks[1:]) / 2
plot_alpha = 0.8

train_data, test_data, divider, _ = load_dataset(filename='data/character_data.pickle')
test_labels, test_qp, test_qv, test_qa, _, _, test_tau, test_m, test_c, test_g = test_data

fig = plt.figure(figsize=(24.0 / 1.54, 8.0 / 1.54), dpi=100)
fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
fig.canvas.set_window_title('NN')
color_i = ["r", "b", "g", "k"]
legend = [mp.patches.Patch(color=color_i[0], label="NN"),
          mp.patches.Patch(color="k", label="Ground Truth")]

y_t_low = np.clip(1.2 * np.min(np.vstack((test_tau, nn_tau)), axis=0), -np.inf, -0.01)
y_t_max = np.clip(1.5 * np.max(np.vstack((test_tau, nn_tau)), axis=0), 0.01, np.inf)

y_m_low = np.clip(1.2 * np.min(np.vstack((test_m, nn_m)), axis=0), -np.inf, -0.01)
y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, nn_m)), axis=0), 0.01, np.inf)

y_c_low = np.clip(1.2 * np.min(np.vstack((test_c, nn_c)), axis=0), -np.inf, -0.01)
y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, nn_c)), axis=0), 0.01, np.inf)

y_g_low = np.clip(1.2 * np.min(np.vstack((test_g, nn_g)), axis=0), -np.inf, -0.01)
y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, nn_g)), axis=0), 0.01, np.inf)

# Plot Torque
ax0 = fig.add_subplot(2, 4, 1)
ax0.set_title(r"Tau")
ax0.text(s="Joint 0", x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center",
         verticalalignment="center", transform=ax0.transAxes)
ax0.set_ylabel("Torque [Nm]")
ax0.get_yaxis().set_label_coords(-0.2, 0.5)
ax0.set_ylim(y_t_low[0], y_t_max[0])
ax0.set_xticks(ticks)
ax0.set_xticklabels(test_labels)
ax0.vlines(divider, y_t_low[0], y_t_max[0], linestyles='--', lw=0.5, alpha=1.)
ax0.set_xlim(divider[0], divider[-1])

ax1 = fig.add_subplot(2, 4, 5)
ax1.text(s="Joint 1", x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
         horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

ax1.text(s=r"A", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
         verticalalignment="center", transform=ax1.transAxes)

ax1.set_ylabel("Torque [Nm]")
ax1.get_yaxis().set_label_coords(-0.2, 0.5)
ax1.set_ylim(y_t_low[1], y_t_max[1])
ax1.set_xticks(ticks)
ax1.set_xticklabels(test_labels)
ax1.vlines(divider, y_t_low[1], y_t_max[1], linestyles='--', lw=0.5, alpha=1.)
ax1.set_xlim(divider[0], divider[-1])

ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

# Plot Ground Truth Torque:
ax0.plot(test_tau[:, 0], color="k")
ax1.plot(test_tau[:, 1], color="k")

# Plot DeLaN Torque:
ax0.plot(nn_tau[:, 0], color=color_i[0], alpha=plot_alpha)
ax1.plot(nn_tau[:, 1], color=color_i[0], alpha=plot_alpha)

# Plot Mass Torque
ax0 = fig.add_subplot(2, 4, 2)
ax0.set_title("H(q)qdd")
ax0.set_ylabel("Torque [Nm]")
ax0.set_ylim(y_m_low[0], y_m_max[0])
ax0.set_xticks(ticks)
ax0.set_xticklabels(test_labels)
ax0.vlines(divider, y_m_low[0], y_m_max[0], linestyles='--', lw=0.5, alpha=1.)
ax0.set_xlim(divider[0], divider[-1])

ax1 = fig.add_subplot(2, 4, 6)
ax1.text(s="b", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
         verticalalignment="center", transform=ax1.transAxes)

ax1.set_ylabel("Torque [Nm]")
ax1.set_ylim(y_m_low[1], y_m_max[1])
ax1.set_xticks(ticks)
ax1.set_xticklabels(test_labels)
ax1.vlines(divider, y_m_low[1], y_m_max[1], linestyles='--', lw=0.5, alpha=1.)
ax1.set_xlim(divider[0], divider[-1])

# Plot Ground Truth Inertial Torque:
ax0.plot(test_m[:, 0], color="k")
ax1.plot(test_m[:, 1], color="k")

# Plot DeLaN Inertial Torque:
ax0.plot(nn_m[:, 0], color=color_i[0], alpha=plot_alpha)
ax1.plot(nn_m[:, 1], color=color_i[0], alpha=plot_alpha)

# Plot Coriolis Torque
ax0 = fig.add_subplot(2, 4, 3)
ax0.set_title("c(q,qdd)")
ax0.set_ylabel("Torque [Nm]")
ax0.set_ylim(y_c_low[0], y_c_max[0])
ax0.set_xticks(ticks)
ax0.set_xticklabels(test_labels)
ax0.vlines(divider, y_c_low[0], y_c_max[0], linestyles='--', lw=0.5, alpha=1.)
ax0.set_xlim(divider[0], divider[-1])

ax1 = fig.add_subplot(2, 4, 7)
ax1.text(s="c", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
         verticalalignment="center", transform=ax1.transAxes)

ax1.set_ylabel("Torque [Nm]")
ax1.set_ylim(y_c_low[1], y_c_max[1])
ax1.set_xticks(ticks)
ax1.set_xticklabels(test_labels)
ax1.vlines(divider, y_c_low[1], y_c_max[1], linestyles='--', lw=0.5, alpha=1.)
ax1.set_xlim(divider[0], divider[-1])

# Plot Ground Truth Coriolis & Centrifugal Torque:
ax0.plot(test_c[:, 0], color="k")
ax1.plot(test_c[:, 1], color="k")

# Plot DeLaN Coriolis & Centrifugal Torque:
ax0.plot(nn_c[:, 0], color=color_i[0], alpha=plot_alpha)
ax1.plot(nn_c[:, 1], color=color_i[0], alpha=plot_alpha)

# Plot Gravity
ax0 = fig.add_subplot(2, 4, 4)
ax0.set_title("g(q)")
ax0.set_ylabel("Torque [Nm]")
ax0.set_ylim(y_g_low[0], y_g_max[0])
ax0.set_xticks(ticks)
ax0.set_xticklabels(test_labels)
ax0.vlines(divider, y_g_low[0], y_g_max[0], linestyles='--', lw=0.5, alpha=1.)
ax0.set_xlim(divider[0], divider[-1])

ax1 = fig.add_subplot(2, 4, 8)
ax1.text(s="d", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
         verticalalignment="center", transform=ax1.transAxes)

ax1.set_ylabel("Torque [Nm]")
ax1.set_ylim(y_g_low[1], y_g_max[1])
ax1.set_xticks(ticks)
ax1.set_xticklabels(test_labels)
ax1.vlines(divider, y_g_low[1], y_g_max[1], linestyles='--', lw=0.5, alpha=1.)
ax1.set_xlim(divider[0], divider[-1])

# Plot Ground Truth Gravity Torque:
ax0.plot(test_g[:, 0], color="k")
ax1.plot(test_g[:, 1], color="k")

# Plot DeLaN Gravity Torque:
ax0.plot(nn_g[:, 0], color=color_i[0], alpha=plot_alpha)
ax1.plot(nn_g[:, 1], color=color_i[0], alpha=plot_alpha)

plt.show()

fig.savefig(f"figures/nn{epoch_limit}.pdf", format="pdf")
fig.savefig(f"figures/nn{epoch_limit}.png", format="png")
