import cv2
import numpy as np
import time
import os
import sys
import traceback
import logging
from paddleocr import PaddleOCR
from ktp_extractor import KTPExtractor, format_to_target_json
from sim_extractor import SIMExtractor, format_sim_to_json
from image_preprocessor import StandardPreprocessor, SmartSIMPreprocessor

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_logs")
os.makedirs(LOG_DIR, exist_ok=True)

processing_logger = logging.getLogger("document_processor")
processing_logger.setLevel(logging.INFO)
if not processing_logger.handlers:
    _handler = logging.FileHandler(
        os.path.join(LOG_DIR, "processing.log"), encoding="utf-8"
    )
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    processing_logger.addHandler(_handler)
    processing_logger.propagate = False


def identify_document_type(ocr_texts):
    full_text = " ".join(ocr_texts).upper()
    if "SURAT IZIN MENGEMUDI" in full_text or "DRIVING LICENSE" in full_text:
        return "SIM"
    if "PROVINSI" in full_text and (
            "NIK" in full_text or "KARTU TANDA" in full_text):
        return "KTP"
    return "UNKNOWN"


class DocumentProcessor:
    def __init__(self, debug=False):
        print("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='id',
            enable_mkldnn=True
        )

        self.ktp_extractor = KTPExtractor()
        self.sim_extractor = SIMExtractor()

        self.debug = debug
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)

        self.std_preprocessor = StandardPreprocessor(
            debug=self.debug,
            debug_dir=f"{self.debug_dir}/preprocess_std"
        )
        self.smart_preprocessor = SmartSIMPreprocessor(
            debug=self.debug,
            debug_dir=f"{self.debug_dir}/preprocess_smart"
        )
        print("Engine ready.")
        sys.stdout.flush()

    def calculate_sim_completeness(self, data):
        if not data:
            return 0
        score = 0
        if data.get('Nama'):
            score += 1.5
        if data.get('Nomor SIM'):
            score += 1
        if data.get('Tanggal Lahir'):
            score += 1
        addr = data.get('alamat', {})
        if addr and (addr.get('kabupaten') or addr.get('name')):
            score += 1
        if data.get('Pekerjaan') or data.get('Berlaku Sampai'):
            score += 0.5
        return score

    def merge_sim_data(self, primary_data, fallback_data):
        if not primary_data:
            return fallback_data
        if not fallback_data:
            return primary_data

        merged = primary_data.copy()

        for key in ['Nama', 'Nomor SIM', 'Tempat Lahir', 'Tanggal Lahir',
                    'Jenis Kelamin', 'Pekerjaan', 'Berlaku Sampai']:
            if not merged.get(key) and fallback_data.get(key):
                merged[key] = fallback_data[key]

        addr_prim = merged.get('alamat', {})
        addr_fall = fallback_data.get('alamat', {})

        if not isinstance(addr_prim, dict):
            addr_prim = {}
        if not isinstance(addr_fall, dict):
            addr_fall = {}

        merged_addr = addr_prim.copy()
        for k in ['name', 'rt_rw', 'kel_desa', 'kecamatan',
                  'kabupaten', 'provinsi']:
            if not merged_addr.get(k) and addr_fall.get(k):
                merged_addr[k] = addr_fall[k]

        merged['alamat'] = merged_addr
        return merged

    def process_image(self, image_path):
        start_time = time.time()
        filename = os.path.basename(image_path)
        doc_type = "UNKNOWN"
        status = 500
        log_extra = {}

        try:
            image = cv2.imread(image_path)
            if image is None:
                status = 404
                return {
                    "status": 404,
                    "error": True,
                    "message": "Image not found"
                }

            std_image = self.std_preprocessor.preprocess(image)

            if self.debug:
                cv2.imwrite(
                    os.path.join(self.debug_dir, "ocr_input_std.jpg"),
                    std_image
                )

            ocr_result_std = self.ocr.predict(std_image)

            if ocr_result_std and ocr_result_std[0]:
                doc_type = identify_document_type(
                    ocr_result_std[0].get("rec_texts", [])
                )

            if doc_type == "UNKNOWN":
                print("[INFO] Std Preprocessing UNKNOWN. Checking Raw image...")
                ocr_result_raw = self.ocr.predict(image)
                if ocr_result_raw and ocr_result_raw[0]:
                    doc_type = identify_document_type(
                        ocr_result_raw[0].get("rec_texts", [])
                    )
                    if doc_type != "UNKNOWN":
                        ocr_result_std = ocr_result_raw

            sys.stdout.flush()

            if doc_type == "KTP":
                data = self.ktp_extractor.process_ktp(
                    ocr_result_std,
                    return_trace=False
                )
                result = format_to_target_json(data)
                status = result.get("status", 500)
                log_extra["nik"] = data.get("NIK") if data else None
                log_extra["nama"] = data.get("Nama") if data else None
                return result

            if doc_type == "SIM":
                texts = (
                    ocr_result_std[0].get("rec_texts", [])
                    if ocr_result_std and ocr_result_std[0] else []
                )
                sim_version = self.sim_extractor.detect_version(texts)

                data_std = self.sim_extractor.process_sim(ocr_result_std)
                score_std = self.calculate_sim_completeness(data_std)

                print(
                    f"[INFO] SIM: {sim_version}, Std Score: {score_std}"
                )
                sys.stdout.flush()

                if sim_version == "SMART" or score_std < 4.0:
                    try:
                        print("[INFO] Triggering Smart SIM Preprocessor...")
                        sys.stdout.flush()

                        smart_image = self.smart_preprocessor.preprocess(image)
                        if self.debug:
                            cv2.imwrite(
                                os.path.join(self.debug_dir, "ocr_input_smart.jpg"),
                                smart_image
                            )

                        ocr_result_smart = self.ocr.predict(smart_image)
                        data_smart = self.sim_extractor.process_sim(ocr_result_smart)
                        score_smart = self.calculate_sim_completeness(data_smart)

                        print(f"[INFO] Smart Path Score: {score_smart}")
                        sys.stdout.flush()

                        if score_smart >= score_std:
                            final_data = self.merge_sim_data(data_smart, data_std)
                            result = format_sim_to_json(final_data)
                            status = result.get("status", 500)
                            log_extra["nama"] = (
                                final_data.get("Nama") if final_data else None
                            )
                            return result
                    except Exception as e:
                        print(f"[ERROR] Smart SIM Path Failed: {e}")
                        traceback.print_exc()
                        sys.stdout.flush()

                result = format_sim_to_json(data_std)
                status = result.get("status", 500)
                log_extra["nama"] = data_std.get("Nama") if data_std else None
                return result

            status = 400
            return {
                "status": 400,
                "error": True,
                "message": "Unknown document type"
            }

        except Exception as e:
            traceback.print_exc()
            status = 500
            log_extra["error_message"] = str(e)
            return {
                "status": 500,
                "error": True,
                "message": f"Internal Error: {str(e)}"
            }

        finally:
            duration = time.time() - start_time
            log_parts = [
                f"file={filename}",
                f"doc_type={doc_type}",
                f"status={status}",
                f"duration={duration:.2f}s",
            ]
            for key, value in log_extra.items():
                if value:
                    log_parts.append(f"{key}={value}")
            processing_logger.info(" | ".join(log_parts))