#!/bin/bash

nkf -d < unity_simulator_launcher.sh > unity_simulator_launcher.sh.tmp
chmod +x unity_simulator_launcher.sh.tmp
./unity_simulator_launcher.sh.tmp

#unity_simulator_launcher.sh.tmp が正常に実行できることを確認できたら、

#mv unity_simulator_launcher.sh.tmp unity_simulator_launcher.sh
