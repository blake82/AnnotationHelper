
---------------------------------------------------------------------------------------------------------------------
Detection Dataset Integrity Helper

Goal
Enhance Dataset Integrity: This project provides an automated and manual validation pipeline to verify Detection Labels using a Classification Model.

---------------------------------------------------------------------------------------------------------------------
Workflow Overview
The following steps outline the process of validating detection labels by cross-checking them with classification predictions and performing manual GUI reviews.

---------------------------------------------------------------------------------------------------------------------
Step 1: Data Preparation
Step1_Crop_bbox.py: Extracts (crops) objects from the Detection Dataset based on existing Label (BBOX) coordinates and saves them as individual images.

Step 2: Automated Pre-filtering
Classification Model Prediction: Run a Classification Model of your choice on the cropped images.

Step2_Filter_Correct_predictions.py: Automatically removes images where the Classification prediction matches the Detection label. Since these are likely correct, they are excluded to minimize manual workload.

Step 3: Manual GUI Review
Step3_Valid_Check_Cropped_Img.py: A Windows-based GUI tool for manual inspection.

Verify if the Class ID is correct and the BBOX position is accurate.

Selection Options: * SUCCESS: Correct label.

DELETE: Incorrect or invalid object.

MODIFY: Correct object but requires BBOX adjustment.

Step 4: Final Verification & Label Update
Step4_Modify_Check_After_Valid_Check.py: Generates a validation directory based on crop_valid.txt.

Visual cues are added: RED BBOX for DELETE items, ORANGE BBOX for MODIFY items.

Users perform a final check using standard labeling tools.

Crucial: Once finished, only the .txt (label) files should be copied back to the original dataset. Do not overwrite original images.

Step 5: Post-processing & Cleanup
Step5_Move_to_Classfication_and_Delete_Crop.py:

Items marked as DELETE are permanently removed.

The remaining cropped images are backed up for future Classification Model training (Fine-tuning).

Finally, the temporary crop directory is cleaned up to save space.


---------------------------------------------------------------------------------------------------------------------
목적 : Detection Data Set 의 무결성 강화

---------------------------------------------------------------------------------------------------------------------
설명 : Detection 에서 사용하는 Label의 Validation Check를 Classification Model을 활용해 자동으로 검수하는 과정입니다. 

---------------------------------------------------------------------------------------------------------------------

아래의 프로세스를 따라 검수 작업을 수행하세요.


1.  Step1_Crop_bbox.py	: Detection Data Set Label 영역을 Crop Image로 저장
2.  Classification Model Prediction --> Classification Model 선택은 자유
3.  Step2_Filter_Correct_predictions.py : Classification Model 예측과 동일한 Label 은 검수 불필요 하므로 제거
4.  Step3_Valid_Check_Cropped_Img.py : Windows 환경에서 GUI 를 활용해, Class ID 가 동일한지, BBOX 위치가 알맞은지 확인
				     : 삭제 필요시 DELETE 선택 , 수정 필요시 MODIFY 선택 
5.  Step4_Modify_Check_After_Valid_Check.py : 4번 작업을 완료 후 crop_valid.txt 를 기반으로 validation 디렉터리를 만들고, 
				     삭제는 RED, 수정은 ORANGE BBOX 로 표시
					label Img 와 같은 툴로 열어서 사용자가 직접 검수 작업 진행 
					작업이 완료되면, txt 파일만 원본에 복사 (이미지를 덮어쓰면 안됨)
6.  Step5_Move_to_Classfication_and_Delete_Crop.py : 
                                   DELETE 된 Image 는 삭제된 뒤, 남은 CROP 이미지는 추후 Classification Model 에 반영 할 수 있도록  
                                   남겨두고 사용자가 직접 한곳에 백업하여 재 사용 하도록 함. 그 뒤 crop 디렉터리 삭제 

