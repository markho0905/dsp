# RoomCorr 交接說明（給 Claude Code）

更新：2026-11-12 02:30。原始碼在 ~/Desktop/RoomCorr（無 git；本檔也在此資料夾）。使用者以繁體中文溝通；回覆請用中文。
優先順序（使用者指示）：先把軟體的演算法與介面做好，做「通用」的功能；不要針對他房間的特定頻率做研究或調整，報告數據即可。

## 現況
- macOS 房間校正 App：PySide6 + numpy/scipy + sounddevice。量測 → 分析 → 目標曲線 → 濾波器 → 低頻控制 → 匯出，可即時切換校正。
- 聆聽鏈：Roon → BlackHole 2ch → RoomCorr → Audiolab 2.0（Omnia），48 kHz，緩衝 4096。裝置問題已解決。
- 安裝／重建：`bash Install_RoomCorr.command`（建 .venv、PyInstaller 打包到 /Applications/RoomCorr.app，約 5 分鐘）。
  重建前先 `pkill -f RoomCorr.app`。啟動／當機記錄在 ~/Library/Logs/RoomCorr.log。
- 直接以原始碼執行：`.venv/bin/python roomcorr_app.py`。
- 量測資料：~/Library/Application Support/RoomCorr/sessions；設定檔：…/profiles/<標準|夜間|電影>/；QSettings 網域 com.roomcorr.RoomCorr。
- REW 5.40 beta 的 API 開在 http://localhost:4735（/measurements、/measurements/{id}/impulse-response 等），可直接抓 IR 來分析。

## 檔案
- roomcorr.py：量測／FIR 設計核心（CLI：`python roomcorr.py selftest`）
- roomcorr_measure.py：掃頻與量測工作階段；roomcorr_analysis.py：Analysis（分析、報告、濾波器設計、auto_settings）
- roomcorr_dsp.py：rePhase 式區塊（分頻、PEQ、全通、延遲）、MSO／Bass Control 最佳化、低音管理組裝
- roomcorr_tf.py：光譜圖（每個頻率各自的高斯視窗）、瀑布圖（CSD）、decay_times、schroeder_rt、clarity、modal_decay
- roomcorr_firopt.py：FIR 工具的預測模型（Model）與一鍵最佳化（optimise）、評估表（metrics_table）
- roomcorr_engine.py：即時分割卷積引擎（單一程序、sd.Stream duplex）；Profile 載入
- roomcorr_app.py：MainWindow、頂欄、對話框（音訊、量測設定、進階設定）、匯出、自動最佳化
- roomcorr_pages.py：六頁（量測／分析／目標曲線／濾波器／低頻控制／匯出）
- roomcorr_tabs.py：FirToolTab（FIR 工具）、Worker、dspin（舊的 BassTab 已刪除）
- roomcorr_widgets.py：CurveWidget（曲線＋懸停讀值）、SpectrogramWidget、WaterfallWidget（3D）、LevelMeter
- roomcorr_theme.py：主題與自訂元件（Card、Segmented、Toggle、ToggleBanner、CapsuleToggle…）

## 介面（2026-11-12 整理後）
- **頂欄**（每頁共用）：標準／夜間／電影、校正開關、Bypass 按鈕。鍵盤 **B** 在任何頁面與對話框切換 Bypass：
  由 MainWindow.eventFilter 在 **ShortcutOverride** 階段處理（Qt 在交給輸入法之前送出，所以注音開著也有效）；
  焦點在真正的文字欄位（QLineEdit／可編輯 QTextEdit，非數字欄）時放行讓字母輸入。各頁原本重複的切換／Bypass 都已移除。
- **分析**：頻率響應／光譜圖／瀑布圖，可選校正前／後、全頻／低頻 15–500 Hz；量測矩陣；分析報告（含設計結果文字）。
  光譜圖 REW 式色階，圖上方「消散」色帶（降 20 dB 的時間，越久越紅）。瀑布圖為 REW 式 3D 曲面（2026-11-12 02:10 改為 numpy 軟體光柵：
  逐像素 Gouraud 色階、法線打光、Z 緩衝、每片切片描邊、前排與兩側布幕；低頻 61 片×420 點、全頻 41 片；0 ms 在最後方、頻率左低右高）：
  右鍵拖曳旋轉（拖曳中降解析度）、滾輪縮放、雙擊重設。注意深度比較要用 float32（曾因 float64/float32 混用造成整片網紋破洞）。
  校正後＝量測 IR 卷積目前濾波器的預測。計算在 Worker 執行緒，結果快取。
- **目標曲線**：預設／自訂（低頻增強、轉折、高頻傾斜、校正上限）、可拖曳的目標點。
- **濾波器**：相位類型、taps、取樣率、平滑、最大提升／衰減、MIMO、保留低頻能量；按鈕：進階設定…、FIR 工具…、預覽（到分析頁）、自動最佳化並套用。
- **低頻控制**：MSO（多低音砲）／Bass Control（衰減控制、限制最大提升；目標與提升上限改到目標曲線／濾波器頁）。
- **匯出**：資料夾、檔名預覽、確認儲存（同時存成目前模式設定檔並啟用）、存檔前試聽開關。
- **FIR 工具**（濾波器頁 → FIR 工具…）：區塊清單選取即可直接改數值，圖表 40 ms、試聽 400 ms 內更新；區塊可指定左／右／雙聲道；
  幅度／相位／群延遲 ×「預測聽感」／「只看這條鏈」；一鍵最佳化（參數：最大提升／衰減、EQ 數、IIR／線性相位分界、
  左右相位對齊、左右音量對齊、語音加權、低頻衰減加權）；評估表（RMS 偏差、左右差、L+R 一致性、群延遲起伏、
  低頻衰減、T20、C50／D50）；匯出 FIR（_L／_R 兩檔）、套用到設定檔（原檔 .bak）。鏈即時套用在播放上，按 B 比較。
- 所有圖表都有滑鼠懸停讀值。

## 演算法重點
- 音量對齊（2026-11-12 02:30）：Profile.mid_gain＝濾波器 300 Hz–3 kHz 的線性增益；level_match 開啟時引擎在 **Bypass 路徑**乘上
  這個增益（只衰減、不削波），濾波路徑不再提升。Engine.select(prof, ref) 的 ref 是 Bypass 要對齊的設定檔；頂欄 Bypass 按鈕顯示對齊 dB。
- 保留低頻能量（設定 keep_bass，roomcorr.design_filter）：300 Hz 以下以 1/f 加權讓修正量平均為 0（含 1.7 倍八度的過渡），
  目標曲線在低頻同步平移，回報用的 tgt 也跟著移；info['bass_off'] 記錄平移量。離線測試：左低頻平均 8.0→7.0 dB（關閉時 2.2）。
- 一鍵最佳化（roomcorr_firopt.optimise）：每聲道貪婪加 PEQ（f_iir 以下自然相位，以上線性相位），每個再用 L-BFGS-B 微調，最後聯合微調；
  誤差加權：凹陷 0.35、位置間差異大的頻段降權、300 Hz–4 kHz ×1.6、餘音長的頻段 ×1–3；提升碰到上限的區域不再重複加提升；
  相近濾波器合併。左右音量只衰減較大聲的一邊。左右相位：差分演化搜尋延遲（±1.5 ms）＋最多兩個全通（Q ≤ 1.2），
  目標是 30 Hz–1 kHz 的 L+R 合成一致性，群延遲增加有懲罰；全通若讓預測低頻餘音變長 >8% 就只保留延遲。
- ToggleBanner 曾在 paintEvent 內 setStyleSheet 造成無限重繪（GUI 100% CPU，音訊斷音），已改為狀態變化時才 restyle。
- Worker（QThread）曾在 run() 還沒結束時被 Python 回收（擁有者在 done/failed 槽裡把參照設為 None）→ Qt fatal「QThread: Destroyed
  while thread is still running」SIGABRT（2026-11-12 02:09 的崩潰）。現在 Worker 自己放在 Worker._alive 集合，finished 後才移除。

## 測試方法
- 離線 GUI：`QT_QPA_PLATFORM=offscreen`，patch `RA.MainWindow.start_audio = lambda self,*a,**k: None`，`w.save_cfg = lambda: None`；
  等 w.analysis／w.design_result 就緒後操作頁面、`grab().save()` 看圖。按鍵用 `QtTest.QTest.keyClick`。
- 計算 paint 事件數可抓重繪迴圈（每頁靜止時應只有個位數）。
- 實機：computer-use 的背景 AXPress 按不到自訂 NavButton，要用全螢幕滑鼠點擊；每批動作前先 `osascript -e 'tell application "RoomCorr" to activate'`。

## REW 1111 / 1111.2 分析（2026-11-11 量測）
- 1111.2（目前的濾波器）比 1111 好：20–80 Hz 平整度（去斜率標準差）左 3.3→0.8、右 4.0→1.5 dB；前振鈴 −57～−63 dB
  （1111 為 −31 dB）；C50 左 +18.1、右 +18.0 dB。
- 兩版共同：濾波後 20–200 Hz 比 bypass 低 8–10 dB；左右差 20–200 Hz 約 5 dB；左右 500 Hz–2 kHz 音量差約 3 dB（左大聲）；
  低頻 T20 63–125 Hz 約 0.4–0.5 s，其餘約 0.2 s。（只報告，不針對特定頻率調整。）

## 待確認／待辦
- B 鍵在注音輸入法下的實機驗證：離線測試通過，實機因螢幕休眠尚未親眼確認（最後一次重建 01:50 左右）。
- 3D 瀑布圖在實機的大小／流暢度；一鍵最佳化的結果只在離線預測驗證，尚未用 REW 重新量測。
- 低音砲功能沒有真實資料可測。

## MIMO 支援喇叭（2026-11-12，雲端 session）
- 舊版 `Analysis._mimo` 的問題（使用者回報「開 MIMO 後頻響更差」）：它只求解「左＋右總和」在三個量測點平坦（目標振幅 2×、所有位置共用同一個複數目標相位），
  再把解出的兩條濾波器當成左右各自的校正；單獨看任一喇叭時並沒有被校正，而且 300–510 Hz 用複數混合 SISO 與 MIMO 兩個不相干的解，會產生梳狀凹陷。
- 新版（Dirac ART 式）：每個來源聲道保留自己的 SISO 濾波器，另一顆喇叭多一條「支援路徑」（ds.hx["LR"]＝左聲道→右喇叭、["RL"]），
  在每個同時量到 L、R 的位置上以最小平方聯合求解（逐頻點 2×2），目標振幅不超過 SISO 平均已達到的 +3 dB（不硬填共同深谷），
  增益上限用「逐頻點加大正則化直到合規」而不是裁切。帶寬 `mimo_fmax`（預設 150 Hz，進階設定 60–300）、`mimo_support`（支援強度 0–100%）、
  `mimo_mode`：space＝空間一致（目標相位沿用 SISO 的相位，只修每個位置的幅度，預設）、decay＝餘音控制（目標為主喇叭直達聲相位，
  也抵消模態餘音；餘量多耗 3–5 dB，平均曲線可能略差）。
- 引擎本來就支援任意 input→output 路徑（MatrixConv／Profile outputs）；匯出時多兩個檔 `<slot>_L_fromR.wav`、`<slot>_R_fromL.wav`，
  設定檔資料夾改存 profile.json（save_multi_profile 支援每條路徑各自的 FIR：path 4-tuple (input, gain, h, suffix)）。
  試聽／預覽用 MainWindow.design_profile；低音管理 build_bass_system 的 main_h 可含 "LR"/"RL"，交叉路徑套同樣的高通與延遲。
- 顯示／預測都含支援路徑：Analysis.curves_for、_achieved、per[ch].after／stats（pairs 視窗）、AnalysisPage.ir_for（光譜圖／瀑布圖）、
  firopt.Model(hx=…)（FIR 工具的預測與評估表）。
- 測試：`python3 mimo_selftest.py`（合成矩形房間、兩喇叭共用模態、3 個位置；印出 SISO／兩種 MIMO 的每喇叭誤差、位置間差異、餘音）；
  `QT_QPA_PLATFORM=offscreen python3 mimo_gui_check.py --shots 目錄`（真的跑 MainWindow：分析→開 MIMO→匯出→Profile.from_folder→Engine 卷積，
  驗證左聲道單獨輸入時右喇叭在 40–120 Hz 有輸出、400 Hz 以上沒有）。合成房間結果（25–300 Hz）：
  SISO 每喇叭 RMS 7.1 dB／位置間 1.9 dB → 空間一致 6.0／1.5（餘量 −2 dB）→ 餘音控制 5.3／1.1、餘音 −6%（餘量 −5 dB）。
- 主喇叭目前只在位置 1–3（±30 cm）量測；MIMO 的效益隨位置數與分布增加，若要像 Dirac 一樣，量測流程應讓主喇叭也量 npos 個位置（待辦）。
