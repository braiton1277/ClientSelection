import copy
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DEVICE, SEED, log_step, seed_worker
from model import SmallCNN
from data import (
    SwitchableTargetedLabelFlipSubset,
    make_server_val_balanced,
    make_clients_dirichlet_indices,
)
from metrics import eval_acc, eval_loss, gini_coefficient, windowed_reward, dynamic_batch_size
from server import (
    compute_deltas_proj_mom_probe_now_and_fo,
    local_train_selected,
    apply_fedavg,
    update_staleness_streak,
)
from agent import VDNSelector, build_context_matrix_vdn


def run_experiment(
    rounds: int = 300,
    n_clients: int = 50,
    k_select: int = 15,
    dir_alpha: float = 0.3,
    # Ataque
    initial_flip_fraction: float = 0.0,
    flip_add_fraction: float = 0.20,
    attack_rounds: List[int] = None,
    flip_rate_initial: float = 1.0,
    flip_rate_new_attack: float = 1.0,
    targeted_only_map_classes: bool = True,
    target_map: Optional[Dict[int, int]] = None,
    # Treino
    max_per_client: int = 2500,
    local_lr: float = 0.005,
    local_steps: int = 10,        # usado nas métricas (todos os 50 clientes)
    local_epochs: int = 5,        # usado no FedAvg (só os K selecionados)
    run_random: bool = True,
    run_vdn: bool = True,
    probe_batches: int = 5,
    mom_beta: float = 0.90,
    # SGD
    momentum: float = 0.95,
    weight_decay: float = 1e-4,
    nesterov: bool = True,
    # RL
    reward_window_W: int = 5,
    marl_eps: float = 0.15,
    marl_swap_m: int = 2,
    marl_lr: float = 1e-3,
    marl_gamma: float = 0.90,
    marl_hidden: int = 128,
    marl_target_sync_every: int = 20,
    warmup_transitions: int = 200,
    start_train_round: int = 100,
    updates_per_round: int = 50,
    train_every: int = 1,
    buf_size: int = 20000,
    batch_base: int = 64,
    batch_max: int = 256,
    batch_buffer_ratio: int = 4,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    per_beta_steps: int = 4000,
    per_eps: float = 1e-3,
    # Eval
    val_shuffle: bool = False,
    val_per_class: int = 200,
    eval_max_batches: int = 20,
    print_every: int = 10,
    print_advfo_every: int = 20,
    out_dir: str = ".",
):
    if attack_rounds is None:
        attack_rounds = [150]
    attack_rounds = sorted(list(set(int(x) for x in attack_rounds)))

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tfm_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ---------- JSON ----------
    run_id = uuid.uuid4().hex[:10]
    out_path = Path(out_dir) / f"results_random_vs_vdn_seed{SEED}_{run_id}.json"

    log = {
        "meta": {
            "run_id": run_id, "seed": int(SEED), "device": str(DEVICE),
            "rounds": int(rounds), "n_clients": int(n_clients), "k_select": int(k_select),
            "dir_alpha": float(dir_alpha), "mom_beta": float(mom_beta),
            "initial_flip_fraction": float(initial_flip_fraction),
            "flip_add_fraction": float(flip_add_fraction),
            "attack_rounds": list(attack_rounds),
            "flip_rate_initial": float(flip_rate_initial),
            "flip_rate_new_attack": float(flip_rate_new_attack),
            "targeted_only_map_classes": bool(targeted_only_map_classes),
            "target_map": target_map if target_map is not None else "default_pair_swaps",
            "buf_size": int(buf_size), "warmup_transitions": int(warmup_transitions),
            "start_train_round": int(start_train_round),
            "updates_per_round": int(updates_per_round), "train_every": int(train_every),
            "print_advfo_every": int(print_advfo_every),
            "local_steps": local_steps,
            "local_epochs": local_epochs,
            "run_random": run_random,
            "run_vdn": run_vdn,
        },
        "attack_schedule": [],
        "tracks": {
            "random": {"test_acc": [], "selection_count_total_per_client": [0] * n_clients, "selection_phases": []},
            "vdn":    {"test_acc": [], "selection_count_total_per_client": [0] * n_clients, "selection_phases": []},
        },
    }

    def save_json():
        for key in ["random", "vdn"]:
            cnt = np.array(log["tracks"][key]["selection_count_total_per_client"], dtype=np.int64)
            log["tracks"][key]["final_metrics"] = {
                "gini_selection_total": float(gini_coefficient(cnt)),
                "total_selections": int(cnt.sum()),
            }
            for ph in log["tracks"][key]["selection_phases"]:
                if ph.get("end_round") is None:
                    ph["end_round"] = int(rounds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"\n[JSON] salvo em: {str(out_path)}\n", flush=True)

    def start_phase(track: str, start_round: int, attacked: List[int]):
        log["tracks"][track]["selection_phases"].append({
            "start_round": int(start_round), "end_round": None,
            "attacked_clients_snapshot": list(attacked),
            "selection_count_per_client": [0] * n_clients,
        })

    def bump(track: str, selected: List[int]):
        total = log["tracks"][track]["selection_count_total_per_client"]
        phase = log["tracks"][track]["selection_phases"][-1]["selection_count_per_client"]
        for i in selected:
            total[i] += 1
            phase[i] += 1

    # ---------- Dados ----------
    log_step("Carregando CIFAR-10...")

    train_ds      = datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm_train)
    train_ds_eval = datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm_eval)
    test_ds       = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_eval)

    server_val_idxs = make_server_val_balanced(train_ds_eval, per_class=val_per_class, seed=SEED + 4242)
    server_val_set  = set(server_val_idxs)
    train_pool_idxs = [int(i) for i in range(len(train_ds)) if int(i) not in server_val_set]

    train_pool      = Subset(train_ds,      train_pool_idxs)
    train_pool_eval = Subset(train_ds_eval, train_pool_idxs)

    g_val = torch.Generator()
    g_val.manual_seed(SEED + 123)
    val_loader  = DataLoader(Subset(train_ds_eval, server_val_idxs), batch_size=256,
                             shuffle=val_shuffle, generator=g_val, worker_init_fn=seed_worker, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    log_step(f"Gerando split Dirichlet (alpha={dir_alpha}) para {n_clients} clientes...")
    client_idxs = make_clients_dirichlet_indices(
        train_pool, n_clients=n_clients, alpha=dir_alpha, seed=SEED + 777
    )

    # ---------- Ataque inicial ----------
    n_init = int(round(initial_flip_fraction * n_clients))
    rng_init = np.random.RandomState(SEED + 999)
    attacked_set = set(
        rng_init.choice(np.arange(n_clients), size=n_init, replace=False).tolist()
    ) if n_init > 0 else set()

    attack_rate_per_client = np.zeros(n_clients, dtype=np.float32)
    for cid in attacked_set:
        attack_rate_per_client[cid] = float(flip_rate_initial)

    # ---------- Loaders dos clientes ----------
    client_train_loaders: List[DataLoader] = []
    client_eval_loaders:  List[DataLoader] = []
    client_sizes: List[int] = []
    switchable_ds: List[SwitchableTargetedLabelFlipSubset] = []
    switchable_ds_eval: List[SwitchableTargetedLabelFlipSubset] = []

    g_train = torch.Generator()
    g_train.manual_seed(SEED + 10001)

    for cid, idxs in enumerate(client_idxs):
        if max_per_client is not None:
            idxs = idxs[:max_per_client]
        client_sizes.append(len(idxs))

        ds_c_train = SwitchableTargetedLabelFlipSubset(
            base_ds=train_pool, indices=idxs, n_classes=10,
            seed=SEED + 1000 + cid,
            enabled=(cid in attacked_set),
            attack_rate=float(attack_rate_per_client[cid]),
            target_map=target_map,
            only_map_classes=targeted_only_map_classes,
        )

        ds_c_eval = SwitchableTargetedLabelFlipSubset(
            base_ds=train_pool_eval, indices=idxs, n_classes=10,
            seed=SEED + 1000 + cid,
            enabled=(cid in attacked_set),
            attack_rate=float(attack_rate_per_client[cid]),
            target_map=target_map,
            only_map_classes=targeted_only_map_classes,
        )

        switchable_ds.append(ds_c_train)
        switchable_ds_eval.append(ds_c_eval)

        client_train_loaders.append(DataLoader(ds_c_train, batch_size=32, shuffle=True,
                                               generator=g_train, worker_init_fn=seed_worker, num_workers=0))
        client_eval_loaders.append(DataLoader(ds_c_eval, batch_size=32, shuffle=False, num_workers=0))

    # ---------- Modelos ----------
    base = SmallCNN().to(DEVICE)

    if run_random:
        model_rand   = copy.deepcopy(base).to(DEVICE)
        rng_rand_sel = random.Random(SEED + 424242)
        start_phase("random", 1, sorted(list(attacked_set)))

    if run_vdn:
        model_vdn   = copy.deepcopy(base).to(DEVICE)
        staleness_v = np.zeros(n_clients, dtype=np.float32)
        streak_v    = np.zeros(n_clients, dtype=np.int32)
        loss_hist_v: List[float] = []
        pending_v:   Optional[Tuple] = None
        mom_v:       Optional[torch.Tensor] = None

        agent_v = VDNSelector(
            n_agents=n_clients, d_in=5, k_select=k_select, hidden=marl_hidden,
            lr=marl_lr, weight_decay=1e-4, gamma=marl_gamma, grad_clip=1.0,
            target_sync_every=marl_target_sync_every, buf_size=buf_size,
            batch_size=batch_base, train_steps=max(1, updates_per_round),
            per_alpha=per_alpha, per_beta_start=per_beta_start,
            per_beta_end=per_beta_end, per_beta_steps=per_beta_steps,
            per_eps=per_eps, double_dqn=True, seed=SEED + 10,
        )
        start_phase("vdn", 1, sorted(list(attacked_set)))

    print(f"\nDEVICE={DEVICE} | N_CLIENTS={n_clients} | K={k_select} | rounds={rounds}")
    print(f"dir_alpha={dir_alpha} | attacked_init={n_init} | local_steps={local_steps}(metrics) epochs={local_epochs}(fedavg)")
    print(f"run_random={run_random} | run_vdn={run_vdn}")
    print(f"Avg client size ~ {np.mean(client_sizes):.1f} samples\n")

    try:
        for t in range(1, rounds + 1):
            log_step(f"\n[round {t}/{rounds}]")

            # ===== Ataque acumulativo =====
            if t in attack_rounds:
                n_add = int(round(flip_add_fraction * n_clients))
                candidates = [i for i in range(n_clients) if i not in attacked_set]
                rng_add = np.random.RandomState(SEED + 5000 + t)
                rng_add.shuffle(candidates)
                add_now = candidates[:min(n_add, len(candidates))]
                for cid in add_now:
                    attacked_set.add(cid)
                    attack_rate_per_client[cid] = float(flip_rate_new_attack)
                for cid in range(n_clients):
                    switchable_ds[cid].set_attack(cid in attacked_set, float(attack_rate_per_client[cid]))
                    switchable_ds_eval[cid].set_attack(cid in attacked_set, float(attack_rate_per_client[cid]))
                log["attack_schedule"].append({
                    "round": int(t), "added_clients": list(map(int, add_now)),
                    "rate_for_added": float(flip_rate_new_attack),
                    "attacked_total_after": int(len(attacked_set)),
                })
                log_step(f"  >>> ATTACK ADD: +{len(add_now)} | total={len(attacked_set)}")

            round_seed = SEED + 50000 + t
            g_train.manual_seed(round_seed)

            # ============================================================
            # TRACK A: RANDOM
            # ============================================================
            if run_random:
                a_rand = eval_acc(model_rand, test_loader, max_batches=80)

                # métricas com steps (rápido, todos os clientes)
                deltas_r, _, _, _, _ = compute_deltas_proj_mom_probe_now_and_fo(
                    model_rand, client_train_loaders, client_eval_loaders, val_loader,
                    local_lr, local_steps, probe_batches=probe_batches,
                    mom=None, mom_beta=mom_beta, round_seed=round_seed + 1,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                )

                K = min(k_select, n_clients)
                sel_r = rng_rand_sel.sample(range(n_clients), K)

                # FedAvg com epochs completos (só os selecionados)
                deltas_r_full = local_train_selected(
                    model_rand, client_train_loaders, sel_r,
                    lr=local_lr, epochs=local_epochs,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                )
                apply_fedavg(model_rand, deltas_r_full, sel_r)
                bump("random", sel_r)
                log["tracks"]["random"]["test_acc"].append(float(a_rand))

            # ============================================================
            # TRACK B: VDN
            # ============================================================
            if run_vdn:
                acc_v = eval_acc(model_vdn, test_loader, max_batches=80)
                _l_before = eval_loss(model_vdn, val_loader, max_batches=eval_max_batches)

                # métricas com steps (rápido, todos os clientes)
                deltas_v, proj_mom_v, probe_now_v, fo_v, mom_v = compute_deltas_proj_mom_probe_now_and_fo(
                    model_vdn, client_train_loaders, client_eval_loaders, val_loader,
                    local_lr, local_steps, probe_batches=probe_batches,
                    mom=mom_v, mom_beta=mom_beta, round_seed=round_seed + 2,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                )

                obs_v = build_context_matrix_vdn(proj_mom_v, probe_now_v, staleness_v, streak_v)

                if pending_v is not None:
                    o_prev, a_prev, r_prev = pending_v
                    agent_v.add_transition(obs=o_prev, act=a_prev, r=r_prev, obs2=obs_v, done=False)

                force_rand = (agent_v.buf.n < warmup_transitions)
                act_v, sel_v = agent_v.select_topk_actions(
                    obs=obs_v, eps=marl_eps, swap_m=marl_swap_m, force_random=force_rand
                )

                q_all = agent_v.q_values(obs_v)
                print("\n[SELECTED DEBUG] cid | flag | state | Q0 Q1")
                for cid in sel_v:
                    flag = "ATTACKER" if cid in attacked_set else "HONEST"
                    st = obs_v[cid]
                    q0, q1 = float(q_all[cid, 0]), float(q_all[cid, 1])
                    print(f"  {cid:02d} | {flag:8s} | [{st[0]:.3f}, {st[1]:+.4f}, {st[2]:.4f}, "
                          f"{st[3]:.3f}, {st[4]:.3f}] | {q0:+.4f} {q1:+.4f}")
                print("")

                if print_advfo_every and t % print_advfo_every == 0:
                    adv = (q_all[:, 1] - q_all[:, 0]).astype(np.float32)
                    order = np.argsort(-adv)
                    print(f"[ADV/FO @ {t}] cid | flag | adv | FO")
                    for cid in order.tolist():
                        flag = "ATTACKER" if cid in attacked_set else "HONEST"
                        print(f"  {cid:02d} | {flag:8s} | adv={adv[cid]:+.6f} | FO={float(fo_v[cid]):+.6f}")
                    print("")

                # FedAvg com epochs completos (só os selecionados)
                deltas_v_full = local_train_selected(
                    model_vdn, client_train_loaders, sel_v,
                    lr=local_lr, epochs=local_epochs,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                )
                apply_fedavg(model_vdn, deltas_v_full, sel_v)
                update_staleness_streak(staleness_v, streak_v, sel_v)

                l_after = eval_loss(model_vdn, val_loader, max_batches=eval_max_batches)
                loss_hist_v.append(l_after)
                r_v = windowed_reward(loss_hist_v[:-1], l_after, W=reward_window_W)
                pending_v = (obs_v.copy(), act_v.copy(), float(r_v))

                trained = False
                if (t >= start_train_round) and (t % train_every == 0) and (agent_v.buf.n >= batch_base) and not force_rand:
                    bs = dynamic_batch_size(agent_v.buf.n, base=batch_base, max_bs=batch_max, ratio=batch_buffer_ratio)
                    agent_v.train(batch_size=bs, train_steps=updates_per_round)
                    trained = True

                bump("vdn", sel_v)
                log["tracks"]["vdn"]["test_acc"].append(float(acc_v))

            # ============================================================
            # PRINT SUMMARY
            # ============================================================
            if t % print_every == 0:
                rand_str    = f"RANDOM={a_rand*100:.2f}%" if run_random else "RANDOM=OFF"
                vdn_str     = f"VDN={acc_v*100:.2f}%"    if run_vdn   else "VDN=OFF"
                buf_str     = f"buf={agent_v.buf.n}"      if run_vdn   else ""
                trained_str = f"trained={int(trained)}"   if run_vdn   else ""
                print(
                    f"[summary @ {t:3d}] {rand_str} | {vdn_str} | "
                    f"attacked={len(attacked_set)} | {buf_str} | {trained_str}",
                    flush=True,
                )

        if run_vdn and pending_v is not None:
            o_prev, a_prev, r_prev = pending_v
            agent_v.add_transition(obs=o_prev, act=a_prev, r=r_prev, obs2=o_prev, done=True)

        print("\nDone.")

    finally:
        save_json()
