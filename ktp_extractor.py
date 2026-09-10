import re
import numpy as np
from thefuzz import process, fuzz
from wilayah_lookup import WilayahLookup


class KTPExtractor:
    def __init__(self):
        self.wilayah = WilayahLookup()

        self.canonical_fields = [
            "PROVINSI", "KABUPATEN", "NIK", "Nama", "Tempat/Tgl Lahir",
            "Jenis Kelamin", "Gol. Darah", "Alamat", "RT/RW", "Kel/Desa",
            "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan",
            "Kewarganegaraan", "Berlaku Hingga"
        ]

        self.truncated_key_map = {
            "RTIRW": "RT/RW",
            "RTRW": "RT/RW",
            "RT.RW": "RT/RW",
            "RIRW": "RT/RW",
            "NIS KELAMIN": "Jenis Kelamin",
            "ENIS KELAMIN": "Jenis Kelamin",
            "EMPAT/TGL": "Tempat/Tgl Lahir",
            "MPAT/TGL": "Tempat/Tgl Lahir",
            "GAMA": "Agama",
            "KERJAAN": "Pekerjaan",
            "PEKARJAON": "Pekerjaan",
            "ATUS PERKAWINAN": "Status Perkawinan",
            "KAL/DESA": "Kel/Desa",
            "KEL/DESA": "Kel/Desa",
            "KACAMUTAN": "Kecamatan",
            "NO KTP": "NIK"
        }

        self.known_values = {
            "Agama": [
                "ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU",
                "CHRISTIAN", "CATHOLIC"
            ],
            "Jenis Kelamin": [
                "LAKI-LAKI", "PEREMPUAN", "LAKI", "PEREMPUAN",
                "MALE", "FEMALE"
            ],
            "Status Perkawinan": [
                "BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI",
                "MARRIED", "SINGLE", "DIVORCED"
            ],
            "Kewarganegaraan": ["WNI", "WNA"],
            "PROVINSI": [
                "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU",
                "KEPULAUAN RIAU", "JAMBI", "SUMATERA SELATAN",
                "KEPULAUAN BANGKA BELITUNG", "BENGKULU", "LAMPUNG",
                "DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH", "DI YOGYAKARTA",
                "JAWA TIMUR", "BANTEN", "BALI", "NUSA TENGGARA BARAT",
                "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT",
                "KALIMANTAN TENGAH", "KALIMANTAN SELATAN",
                "KALIMANTAN TIMUR", "KALIMANTAN UTARA", "SULAWESI UTARA",
                "SULAWESI TENGAH", "SULAWESI SELATAN", "SULAWESI TENGGARA",
                "GORONTALO", "SULAWESI BARAT", "MALUKU", "MALUKU UTARA",
                "PAPUA", "PAPUA BARAT", "PAPUA TENGAH", "PAPUA PEGUNUNGAN",
                "PAPUA SELATAN", "PAPUA BARAT DAYA"
            ]
        }

    def _get_y_center(self, item):
        box = item['box']
        return (box[0][1] + box[3][1]) / 2

    def _looks_like_key(self, text_upper, matched_field):
        """A short OCR line can fuzzy-match a canonical field name via
        partial_ratio purely by coincidence (e.g. value text "AMAT FAOZI"
        partial-matches "Nama" through the shared "AMA" substring), which
        would wrongly reclassify a value line as a key. Guard against that:
        accept short/near-length matches outright, and for longer text only
        accept it as a genuine "Key: Value" combined box when the text
        actually starts with (a fuzzy version of) the field name."""
        field_upper = matched_field.upper()
        if len(text_upper) <= len(field_upper) + 4:
            return True
        prefix = text_upper[:len(field_upper) + 3]
        return fuzz.ratio(prefix, field_upper) > 70

    def _find_second_line(self, recognized_data, key_item, first_line_item,
                           lower_bound_y, claimed_value_ids, key_ids,
                           extra_exclude=None):
        """Find a wrapped second line of text sitting just below
        first_line_item, bounded below by the y-position of the next
        field's key (lower_bound_y) so it doesn't swallow unrelated rows."""
        line1_y = self._get_y_center(first_line_item)

        candidates = []
        for val_item in recognized_data:
            if val_item['id'] in claimed_value_ids:
                continue
            if val_item['id'] in (first_line_item['id'], key_item['id']):
                continue
            if val_item['id'] in key_ids:
                continue

            val_y = self._get_y_center(val_item)
            is_below = val_y > (line1_y + 10)
            is_above_bound = val_y < (lower_bound_y - 10)
            is_close = (val_y - line1_y) < 45

            if not (is_below and is_above_bound and is_close):
                continue

            if extra_exclude and extra_exclude(val_item):
                continue

            candidates.append(val_item)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c['box'][0][1])
        return candidates[0]

    def process_ktp(self, ocr_result, return_trace=False):
        if not ocr_result or not ocr_result[0]:
            return (None, None, None) if return_trace else None

        result_dict = ocr_result[0]
        if not result_dict or not isinstance(result_dict, dict):
            return (None, None, None) if return_trace else None

        boxes = result_dict.get('dt_polys', [])
        texts = result_dict.get('rec_texts', [])
        scores = result_dict.get('rec_scores', [])

        if not texts:
            return (None, None, None) if return_trace else None

        recognized_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            conf = scores[i] if i < len(scores) else 0.0
            recognized_data.append({
                'id': i,
                'box': np.array(box).astype(np.int32),
                'text': text,
                'confidence': conf
            })

        filtered_data = self.filter_spatial_outliers(recognized_data)

        structured_data, trace_info = self.post_process(filtered_data)

        cleaned_data = self.cleanup_data(structured_data)

        if return_trace:
            return cleaned_data, filtered_data, trace_info
        return cleaned_data

    def filter_spatial_outliers(self, recognized_data):
        key_y_positions = []
        for item in recognized_data:
            text_upper = item['text'].upper()
            match, score = process.extractOne(
                text_upper, self.canonical_fields, scorer=fuzz.partial_ratio
            )
            if score > 85:
                key_y_positions.append(self._get_y_center(item))

        if not key_y_positions:
            return recognized_data

        min_y = min(key_y_positions)
        max_y = max(key_y_positions)

        active_height = max_y - min_y
        cutoff_y_bottom = max_y + (active_height * 0.45)
        cutoff_y_top = min_y - (active_height * 0.3)

        filtered = [
            item for item in recognized_data
            if cutoff_y_top <= self._get_y_center(item) <= cutoff_y_bottom
        ]
        return filtered

    def post_process(self, recognized_data):
        potential_keys = []
        potential_values = []
        trace_info = {}
        
        for item in recognized_data:
            text_raw = item['text'].strip()
            text_upper = text_raw.upper()

            if len(text_raw) < 2 and text_raw not in [":", "-"]:
                potential_values.append(item)
                continue

            best_match, score = process.extractOne(
                text_raw, self.canonical_fields, scorer=fuzz.partial_ratio
            )

            truncated_match = None
            for bad_key, correct_key in self.truncated_key_map.items():
                if bad_key in text_upper:
                    truncated_match = correct_key
                    break

            is_key = False

            if truncated_match:
                item['canonical_field'] = truncated_match
                potential_keys.append(item)
                is_key = True
            elif score > 80 and self._looks_like_key(text_upper, best_match):
                item['canonical_field'] = best_match
                potential_keys.append(item)
                is_key = True

            if not is_key:
                potential_values.append(item)

        potential_keys.sort(key=self._get_y_center)
        key_ids = {k['id'] for k in potential_keys}
        key_map = {k['canonical_field']: k for k in potential_keys}

        extracted_data = {}
        claimed_value_ids = set()

        for key_item in potential_keys:
            key_name = key_item['canonical_field']

            if key_name in extracted_data:
                continue

            if key_name in ["PROVINSI", "KABUPATEN"]:
                if key_name == "PROVINSI":
                    value = re.sub(
                        r'PROVINSI', '', key_item['text'], flags=re.IGNORECASE
                    ).strip()
                else:
                    # KABUPATEN keeps its KOTA/KABUPATEN prefix intact -
                    # that word is significant (Kota Bekasi vs Kabupaten
                    # Bekasi are different regions) and is only stripped
                    # later, once resolve_wilayah has used it to pick the
                    # right one.
                    value = key_item['text'].strip()

                value = re.sub(r'^[:\-\.\s]+', '', value).strip()

                if value:
                    extracted_data[key_name] = value
                    trace_info[key_name] = {
                        "value": value,
                        "source_ids": [key_item['id']],
                        "method": "header_strip"
                    }
                    continue

            key_part_match = process.extractOne(key_name, [key_item['text']], scorer=fuzz.partial_ratio)
            
            inline_candidate = ""
            if key_part_match and key_part_match[1] > 70:
                clean_key_text = key_item['text']
                parts = re.split(r'[:]', clean_key_text, maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    inline_candidate = parts[1].strip()
                else:
                    if len(clean_key_text) > len(key_name) + 2:
                        potential_inline = clean_key_text[len(key_name):].strip()
                        if re.match(r'^[:\-\.\s]*', potential_inline):
                             inline_candidate = re.sub(r'^[:\-\.\s]*', '', potential_inline)

            if inline_candidate and len(inline_candidate) > 2:
                extracted_data[key_name] = inline_candidate
                trace_info[key_name] = {
                    "value": inline_candidate,
                    "source_ids": [key_item['id']],
                    "method": "inline_extraction"
                }
                continue

            key_y_center = self._get_y_center(key_item)
            key_x_end = key_item['box'][1][0]
            same_line_candidates = []

            vertical_threshold = 25 

            for val_item in potential_values:
                if val_item['id'] in claimed_value_ids:
                    continue

                val_y_center = self._get_y_center(val_item)
                val_x_start = val_item['box'][0][0]

                if (abs(val_y_center - key_y_center) < vertical_threshold and
                        val_x_start > (key_x_end - 20)):
                    
                    x_dist = val_x_start - key_x_end
                    y_diff = abs(val_y_center - key_y_center)
                    
                    score = x_dist + (y_diff * 15)
                    same_line_candidates.append((score, val_item))

            if same_line_candidates:
                same_line_candidates.sort(key=lambda c: c[0])

                valid_candidates = [
                    c for c in same_line_candidates
                    if not re.match(r'^[:\-\.\s]+$', c[1]['text'])
                ]

                if key_name == "NIK":
                    valid_candidates = [
                        c for c in valid_candidates
                        if re.match(
                            r'^\d',
                            c[1]['text'].replace(' ', '').replace(':', '')
                        )
                    ]

                if valid_candidates:
                    best_candidate = valid_candidates[0][1]
                    value_text = best_candidate['text']
                    used_ids = [best_candidate['id']]
                    method = "geometric_match"

                    if key_name == 'Alamat':
                        rt_rw_key = key_map.get('RT/RW')
                        rt_rw_y = (
                            self._get_y_center(rt_rw_key)
                            if rt_rw_key else float('inf')
                        )

                        def alamat_exclude(val_item):
                            txt_upper = val_item['text'].upper()
                            if re.search(r'\d{3}[/\s-]+\d{3}', val_item['text']):
                                return True
                            if "RT" in txt_upper and "RW" in txt_upper:
                                return True
                            if "KEL/DESA" in txt_upper:
                                return True
                            return False

                        second_line = self._find_second_line(
                            recognized_data, key_item, best_candidate,
                            rt_rw_y, claimed_value_ids, key_ids,
                            extra_exclude=alamat_exclude
                        )
                        if second_line:
                            value_text += f" {second_line['text']}"
                            claimed_value_ids.add(second_line['id'])
                            used_ids.append(second_line['id'])
                            method = "geometric_match_multiline"

                    elif key_name == 'Nama':
                        ttl_key = key_map.get('Tempat/Tgl Lahir')
                        ttl_y = (
                            self._get_y_center(ttl_key)
                            if ttl_key else float('inf')
                        )

                        second_line = self._find_second_line(
                            recognized_data, key_item, best_candidate,
                            ttl_y, claimed_value_ids, key_ids
                        )
                        if second_line:
                            value_text += f" {second_line['text']}"
                            claimed_value_ids.add(second_line['id'])
                            used_ids.append(second_line['id'])
                            method = "geometric_match_multiline"

                    elif key_name == 'Tempat/Tgl Lahir':
                        jk_key = key_map.get('Jenis Kelamin')
                        jk_y = (
                            self._get_y_center(jk_key)
                            if jk_key else float('inf')
                        )

                        second_line = self._find_second_line(
                            recognized_data, key_item, best_candidate,
                            jk_y, claimed_value_ids, key_ids
                        )
                        if second_line:
                            value_text += f" {second_line['text']}"
                            claimed_value_ids.add(second_line['id'])
                            used_ids.append(second_line['id'])
                            method = "geometric_match_multiline"

                    extracted_data[key_name] = value_text
                    claimed_value_ids.add(best_candidate['id'])

                    trace_info[key_name] = {
                        "value": value_text,
                        "source_ids": used_ids,
                        "key_id_used": key_item['id'],
                        "method": method
                    }
            
            if key_name == "NIK" and key_name not in extracted_data:
                below_candidates = []
                for val_item in potential_values:
                    if val_item['id'] in claimed_value_ids: continue
                    val_y_center = self._get_y_center(val_item)
                    y_diff = val_y_center - key_y_center
                    if 0 < y_diff < 50:
                        clean_val = val_item['text'].replace(" ", "").replace(":","")
                        if re.match(r'\d+', clean_val):
                            below_candidates.append(val_item)
                
                if below_candidates:
                    below_candidates.sort(key=lambda x: x['box'][0][1])
                    best_nik = below_candidates[0]
                    extracted_data["NIK"] = best_nik['text']
                    claimed_value_ids.add(best_nik['id'])
                    trace_info["NIK"] = {
                        "value": best_nik['text'],
                        "source_ids": [best_nik['id']],
                        "method": "geometric_below_fallback"
                    }

        self.recover_missing_fields(
            extracted_data, potential_values, claimed_value_ids,
            key_map, trace_info
        )

        return {
            field: extracted_data.get(field)
            for field in self.canonical_fields if extracted_data.get(field)
        }, trace_info

    def recover_missing_fields(self, extracted, values, claimed_ids,
                               key_map, trace_info):

        if "KABUPATEN" not in extracted:
            provinsi_key = key_map.get("PROVINSI")
            nik_key = key_map.get("NIK")

            if provinsi_key:
                y_min = provinsi_key['box'][3][1]
                y_max = (
                    self._get_y_center(nik_key) if nik_key else y_min + 70
                )

                candidates = [
                    val_item for val_item in values
                    if val_item['id'] not in claimed_ids
                    and val_item['id'] != provinsi_key['id']
                    and y_min < self._get_y_center(val_item) < y_max
                ]

                if candidates:
                    candidates.sort(key=self._get_y_center)
                    chosen = candidates[0]
                    # KOTA/KABUPATEN is kept here (not stripped) so
                    # resolve_wilayah can use it to disambiguate regions
                    # that exist as both a city and a regency, e.g. Kota
                    # Bekasi vs Kabupaten Bekasi.
                    value = re.sub(
                        r'^[:\-\.\s]+', '', chosen['text']
                    ).strip()

                    if value:
                        extracted["KABUPATEN"] = value
                        claimed_ids.add(chosen['id'])
                        trace_info["KABUPATEN"] = {
                            "value": value,
                            "source_ids": [chosen['id']],
                            "method": "positional_inference_kabupaten"
                        }

        for field, keywords in self.known_values.items():
            if field in extracted:
                continue

            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue

                text_upper = val_item['text'].upper()
                match = process.extractOne(
                    text_upper, keywords, scorer=fuzz.token_set_ratio
                )

                if not match and field == "Jenis Kelamin":
                    if "LAKILAKI" in text_upper:
                        extracted[field] = "LAKI-LAKI"
                        claimed_ids.add(val_item['id'])
                        trace_info[field] = {
                            "value": "LAKI-LAKI",
                            "source_ids": [val_item['id']],
                            "method": "typo_recovery"
                        }
                        continue

                if match and match[1] > 85:
                    extracted[field] = val_item['text']
                    claimed_ids.add(val_item['id'])
                    trace_info[field] = {
                        "value": val_item['text'],
                        "source_ids": [val_item['id']],
                        "method": "value_keyword_recovery"
                    }
                    break
        
        if "Tempat/Tgl Lahir" not in extracted:
            for val_item in values:
                if val_item['id'] in claimed_ids: continue
                
                txt = val_item['text']
                if re.search(r'\d{2}[-\s/]\d{2}[-\s/]\d{4}', txt):
                    if re.search(r'[A-Za-z]{3,}', txt):
                        extracted["Tempat/Tgl Lahir"] = txt
                        claimed_ids.add(val_item['id'])
                        trace_info["Tempat/Tgl Lahir"] = {
                            "value": txt,
                            "source_ids": [val_item['id']],
                            "method": "regex_date_place_recovery"
                        }
                        break

        if "Nama" not in extracted:
            nik_key = key_map.get("NIK")
            ttl_key = key_map.get("Tempat/Tgl Lahir")

            y_min = -1
            y_max = float('inf')

            if nik_key:
                y_min = nik_key['box'][3][1]
            elif "NIK" in extracted and "NIK" in trace_info:
                 pass 

            if ttl_key:
                y_max = ttl_key['box'][0][1]

            candidates = []
            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue

                y_center = self._get_y_center(val_item)

                valid = False
                if y_min != -1 and y_max != float('inf'):
                    if y_min < y_center < y_max:
                        valid = True
                elif y_min != -1:
                    if y_min < y_center < y_min + 70:
                        valid = True
                elif y_max != float('inf'):
                    if y_max - 70 < y_center < y_max:
                        valid = True

                if valid:
                    candidates.append(val_item)

            if candidates:
                candidates.sort(key=lambda c: c['box'][0][0])
                chosen = candidates[0]
                extracted["Nama"] = chosen['text']
                claimed_ids.add(chosen['id'])
                trace_info["Nama"] = {
                    "value": chosen['text'],
                    "source_ids": [chosen['id']],
                    "method": "positional_inference_name"
                }
        
        if "NIK" not in extracted:
            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue
                clean_text = val_item['text'].replace(" ", "").strip()
                if re.match(r'^\d{16}$', clean_text):
                     extracted["NIK"] = clean_text
                     claimed_ids.add(val_item['id'])
                     trace_info["NIK"] = {
                        "value": clean_text,
                        "source_ids": [val_item['id']],
                        "method": "regex_recovery_16_digits"
                    }
                     break

    def cleanup_data(self, data):
        cleaned_data = {}
        for key, value in data.items():
            if value is None:
                continue

            clean_value = value.strip()
            if clean_value.startswith(':'):
                clean_value = clean_value[1:].strip()

            clean_value = re.sub(r',\s*', ', ', clean_value).strip()

            if key == "Agama":
                match, score = process.extractOne(clean_value.upper(), self.known_values['Agama'])
                if score > 70:
                    clean_value = match

            if key == "PROVINSI":
                match, score = process.extractOne(clean_value.upper(), self.known_values['PROVINSI'])
                if score > 85:
                    clean_value = match
            
            if key == "RT/RW":
                nums = re.findall(r'\d+', clean_value)
                if len(nums) >= 2:
                    clean_value = f"{nums[0]}/{nums[1]}"
                elif len(nums) == 1:
                    clean_value = f"{nums[0]}/-"

            if key == "Jenis Kelamin":
                val_upper = clean_value.upper()
                if "LAKI" in val_upper or "MALE" in val_upper:
                    clean_value = "LAKI-LAKI"
                elif "PEREMPUAN" in val_upper or "FEMALE" in val_upper:
                    clean_value = "PEREMPUAN"
                else:
                    if "LAK" in val_upper or "LK" in val_upper:
                        clean_value = "LAKI-LAKI"
                    elif "PER" in val_upper or "PR" in val_upper:
                        clean_value = "PEREMPUAN"

            if key == "Status Perkawinan":
                val_upper = clean_value.upper()
                fuzzy_match, fuzzy_score = process.extractOne(
                    val_upper, self.known_values['Status Perkawinan']
                )
                if fuzzy_score > 75:
                    val_upper = fuzzy_match
                if any(k in val_upper for k in ["BELUM", "SINGLE"]):
                    clean_value = "BELUM KAWIN"
                elif any(k in val_upper for k in ["KAWIN", "MARRIED"]):
                    clean_value = "KAWIN"
                elif "CERAI" in val_upper or "DIVORCED" in val_upper:
                    if "HIDUP" in val_upper:
                        clean_value = "CERAI HIDUP"
                    elif "MATI" in val_upper:
                        clean_value = "CERAI MATI"
                    else:
                        clean_value = "CERAI"

            if key == "Alamat":
                clean_value = re.sub(
                    r'\s+RT.*', '', clean_value, flags=re.IGNORECASE
                ).strip()
                clean_value = re.sub(
                    r'\s+RW.*', '', clean_value, flags=re.IGNORECASE
                ).strip()

            if key == "Pekerjaan":
                clean_value = clean_value.replace("BURUHHARIAN", "BURUH HARIAN")
                match = self.wilayah.match_pekerjaan(clean_value.upper())
                if match:
                    clean_value = match['name']

            cleaned_data[key] = clean_value.upper()

        self.resolve_wilayah(cleaned_data)

        return cleaned_data

    def resolve_wilayah(self, cleaned_data):
        """Correct PROVINSI/KABUPATEN/Kecamatan/Kel-Desa against the wilayah
        db, each level scoped to the parent resolved just before it so OCR
        typos are matched against a small, relevant candidate set."""

        id_prov = None
        id_kab = None
        id_kec = None

        if cleaned_data.get("PROVINSI"):
            match = self.wilayah.match_provinsi(cleaned_data["PROVINSI"])
            if match:
                cleaned_data["PROVINSI"] = match['name']
                id_prov = match['row']['id']

        if cleaned_data.get("KABUPATEN"):
            raw_kab = cleaned_data["KABUPATEN"]

            # A bare name like "BEKASI" is ambiguous between Kota Bekasi and
            # Kabupaten Bekasi - several regions exist as both. The
            # KOTA/KABUPATEN prefix (kept intact until now) tells us which
            # one, so use it to filter the db instead of guessing.
            prefix_match = re.match(
                r'^(KABUPATEN|KOTA)\b\s*', raw_kab, flags=re.IGNORECASE
            )
            tipe = None
            kab_query = raw_kab
            if prefix_match:
                tipe = 1 if prefix_match.group(1).upper() == 'KABUPATEN' else 2
                kab_query = raw_kab[prefix_match.end():].strip()

            match = self.wilayah.match_kabupaten(
                kab_query, id_prov=id_prov, tipe=tipe
            )
            if match:
                cleaned_data["KABUPATEN"] = re.sub(
                    r'^(KABUPATEN|KOTA)\s+(ADMINISTRASI\s+)?', '',
                    match['name'], flags=re.IGNORECASE
                ).strip()
                id_kab = match['row']['id']
                if id_prov is None:
                    id_prov = match['row']['id_prov']
            else:
                cleaned_data["KABUPATEN"] = kab_query

        if cleaned_data.get("Kecamatan"):
            match = self.wilayah.match_kecamatan(
                cleaned_data["Kecamatan"], id_kab=id_kab, id_prov=id_prov
            )
            if match:
                cleaned_data["Kecamatan"] = match['name']
                id_kec = match['row']['id']
                if id_kab is None:
                    id_kab = match['row']['id_kab']
                if id_prov is None:
                    id_prov = match['row']['id_prov']

        if cleaned_data.get("Kel/Desa"):
            match = self.wilayah.match_kelurahan(
                cleaned_data["Kel/Desa"],
                id_kec=id_kec, id_kab=id_kab, id_prov=id_prov
            )
            if match:
                cleaned_data["Kel/Desa"] = match['name']

        return cleaned_data


def format_to_target_json(data):
    tempat_lahir = None
    tgl_lahir = None

    raw_ttl = data.get("Tempat/Tgl Lahir", "")

    if raw_ttl:
        match = re.search(r'^(?P<place>.*?)\s*(?P<date>\d{2}[-\s/]+\d{2}[-\s/]+\d{4})$', raw_ttl)

        if match:
            tempat_lahir = match.group('place').strip().strip(":.,")
            tgl_lahir = match.group('date').strip().replace(' ', '-')
            tgl_lahir = re.sub(r'[\s/]+', '-', tgl_lahir)
        else:
            parts = raw_ttl.split(',', 1)
            tempat_lahir = parts[0].strip().strip(":.,")
            if len(parts) > 1:
                tgl_lahir = parts[1].strip()

    return {
        "status": 200,
        "error": False,
        "message": "KTP OCR Processed Successfully",
        "data": {
            "document_type": "KTP",
            "nomor": data.get("NIK"),
            "nama": data.get("Nama"),
            "tempat_lahir": tempat_lahir,
            "tgl_lahir": tgl_lahir,
            "jenis_kelamin": data.get("Jenis Kelamin"),
            "agama": data.get("Agama"),
            "status_perkawinan": data.get("Status Perkawinan"),
            "pekerjaan": data.get("Pekerjaan"),
            "kewarganegaraan": data.get("Kewarganegaraan"),
            "alamat": {
                "name": data.get("Alamat"),
                "rt_rw": data.get("RT/RW"),
                "kel_desa": data.get("Kel/Desa"),
                "kecamatan": data.get("Kecamatan"),
                "kabupaten": data.get("KABUPATEN"),
                "provinsi": data.get("PROVINSI")
            },
        }
    }