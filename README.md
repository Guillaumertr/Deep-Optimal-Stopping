# 🧠 Deep Optimal Stopping

This notebook explores the problem of **optimal stopping** for a high-dimensional Bermudan Max-Call option using **deep learning techniques**.

We reformulate the classical dynamic programming approach into a data-driven method, training a neural network to approximate the optimal stopping strategy over simulated scenarios.

---

## 📈 Problem Setup

We consider a Bermudan Max-Call option with multiple underlying assets. Each asset evolves under a **risk-neutral Black-Scholes model**:

$$
S_t^{m, i} = S_0^i \cdot \exp\left[(r - \delta_i - \frac{1}{2} \sigma_i^2)t + \sigma_i W_t^i\right]
$$

- $d$ : number of assets  
- $M$ : number of simulated scenarios  
- $N$ : number of time steps  
- $W_t^i$: Brownian motion component for asset $i$

We aim to solve:

$$
\sup_{\tau \in \mathcal{T}} \mathbb{E}\left[ e^{-r \tau} \left(\max_i S_\tau^i - K\right)^+ \right]
$$

Where $\tau$ is a stopping time based on asset trajectories.

---

## 🧪 What’s Inside

### 1. Asset Simulation

We simulate $M$ scenarios of asset paths using geometric Brownian motion with time-discretization. Each scenario corresponds to a different realization of the Brownian motion.

### 2. Deep Learning Approach

We model the stopping decision at each time as a **classification problem**:
- At each time $t_n$, the network decides whether to **continue** or **stop**.
- The model is trained via **backward induction**, similar to dynamic programming.
- The objective is to maximize the expected discounted payoff.

### 3. Evaluation

After training, we:
- Estimate the option price using out-of-sample scenarios.
- Compare the learned stopping policy to benchmarks (e.g., Longstaff-Schwartz or oracle).
- Visualize policy decisions and asset trajectories.

---

## 🧠 Highlights

- ✅ Handles **multi-asset**, high-dimensional problems.
- 📊 Learns an **adaptive exercise strategy** through neural networks.
- 💡 Bridges financial theory and modern ML.

---

## 📌 Requirements

- Python 3.8+
- `numpy`, `torch`, `matplotlib`, `scipy`, `tqdm`
- GPU recommended for training efficiency

---

## 🧾 References

- Becker, Cheridito & Jentzen (2019). *Deep optimal stopping.*

---

## 🧳 Use Case

This notebook is ideal for:
- Quant researchers exploring deep reinforcement learning for pricing.
- Practitioners interested in high-dimensional option pricing.
- Students looking for practical examples of deep learning in finance.

---

## 👤 Author

👨‍💻 Guillaume Routier  
🎓 MSc Probabilité & Finance – École Polytechnique & Sorbonne Université  
📬 [Contact me on LinkedIn](https://www.linkedin.com/in/guillaume-routier/)

---

## 📌 Disclaimer

These notebooks are for educational and demonstrative purposes only. They do not constitute financial advice.

---

## ⭐ If you find this useful...

Feel free to **star** the repo or get in touch for collaboration!
