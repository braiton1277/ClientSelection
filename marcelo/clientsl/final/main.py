from experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        rounds=500,
        n_clients=50,
        k_select=15,
        dir_alpha=0.3,

        initial_flip_fraction=0.4,
        flip_add_fraction=0.0,
        attack_rounds=[600],
        flip_rate_initial=1.0,
        flip_rate_new_attack=0.0,

        targeted_only_map_classes=True,
        target_map=None,

        max_per_client=2500,
        local_lr=0.01,
        local_steps=10,
        probe_batches=10,

        mom_beta=0.90,
        reward_window_W=5,

        marl_eps=0.15,
        marl_swap_m=2,
        marl_lr=1e-3,
        marl_gamma=0.90,
        marl_hidden=128,
        marl_target_sync_every=20,

        warmup_transitions=50,
        start_train_round=50,
        updates_per_round=50,
        train_every=1,

        buf_size=20000,
        batch_base=32,
        batch_max=256,
        batch_buffer_ratio=4,

        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=4000,
        per_eps=1e-3,

        val_shuffle=False,
        val_per_class=200,
        eval_max_batches=20,
        print_every=1,
        print_advfo_every=20,

        out_dir=".",
    )
