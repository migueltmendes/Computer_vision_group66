BOTTLENECK_LIST=(avg_8x8 max_8x8 global_avg global_max 2x2_conv 3x3_conv 5x5_conv linear)
LRS=(0.1 0.01 0.001 0.0001)

MAX_COUNT=$(( ${#BOTTLENECK_LIST[@]} * ${#LRS[@]} * 3))
COUNT=0

echo "Completed $COUNT/$MAX_COUNT configurations."

for BOTTLENECK in "${BOTTLENECK_LIST[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Running with bottleneck=$BOTTLENECK, freeze_features=False, lr=$LR, load_best=True"
        python3 src_stl10/main.py --bottleneck_type "$BOTTLENECK" --lr "$LR" --use_amp --load_best --tag "lr_tuning"

        COUNT=$((COUNT + 1))
        echo "Completed $COUNT/$MAX_COUNT configurations."

        echo "Running with bottleneck=$BOTTLENECK, freeze_features=True, lr=$LR, load_best=True"
        python3 src_stl10/main.py --bottleneck_type "$BOTTLENECK" --freeze_features --lr "$LR" --use_amp --load_best --tag "lr_tuning"

        COUNT=$((COUNT + 1))
        echo "Completed $COUNT/$MAX_COUNT configurations."

        echo "Running with bottleneck=$BOTTLENECK, freeze_features=True, lr=$LR, load_best=False"
        python3 src_stl10/main.py --bottleneck_type "$BOTTLENECK" --freeze_features --lr "$LR" --use_amp --tag "lr_tuning"

        COUNT=$((COUNT + 1))
        echo "Completed $COUNT/$MAX_COUNT configurations."
    done
done