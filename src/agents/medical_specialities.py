# src/agents/medical_specialties.py
from typing import List, Dict
import random


class MedicalSpecialties:
    """Define medical specialties and their expertise areas"""

    SPECIALTIES = {
        # Primary Care
        "Primary Care Physician": {
            "expertise": "general medicine, preventive care, common conditions, initial diagnosis",
            "keywords": ["general", "routine", "checkup", "common", "prevention"]
        },
        "Internal Medicine": {
            "expertise": "adult diseases, complex diagnoses, chronic disease management",
            "keywords": ["adult", "chronic", "complex", "systemic"]
        },

        # Organ System Specialists
        "Cardiologist": {
            "expertise": "heart conditions, cardiovascular disease, hypertension, arrhythmias",
            "keywords": ["heart", "cardiac", "chest pain", "hypertension", "ECG", "blood pressure"]
        },
        "Pulmonologist": {
            "expertise": "lung diseases, respiratory conditions, breathing problems",
            "keywords": ["lung", "breathing", "respiratory", "cough", "dyspnea", "asthma"]
        },
        "Gastroenterologist": {
            "expertise": "digestive system, GI disorders, liver diseases",
            "keywords": ["stomach", "abdomen", "digestive", "liver", "bowel", "nausea"]
        },
        "Nephrologist": {
            "expertise": "kidney diseases, renal function, dialysis, electrolyte disorders",
            "keywords": ["kidney", "renal", "creatinine", "dialysis", "urine"]
        },
        "Neurologist": {
            "expertise": "brain and nervous system disorders, headaches, seizures, stroke",
            "keywords": ["brain", "headache", "seizure", "nervous", "stroke", "neurological"]
        },
        "Endocrinologist": {
            "expertise": "hormonal disorders, diabetes, thyroid diseases, metabolic conditions",
            "keywords": ["diabetes", "thyroid", "hormone", "endocrine", "metabolic"]
        },

        # Surgical Specialists
        "General Surgeon": {
            "expertise": "surgical procedures, trauma, acute abdomen",
            "keywords": ["surgery", "surgical", "trauma", "acute", "operation"]
        },
        "Orthopedic Surgeon": {
            "expertise": "bone and joint disorders, fractures, musculoskeletal conditions",
            "keywords": ["bone", "joint", "fracture", "orthopedic", "musculoskeletal"]
        },

        # Other Specialists
        "Psychiatrist": {
            "expertise": "mental health, psychiatric disorders, behavioral issues",
            "keywords": ["mental", "psychiatric", "depression", "anxiety", "behavioral"]
        },
        "Dermatologist": {
            "expertise": "skin conditions, rashes, skin cancer",
            "keywords": ["skin", "rash", "dermatology", "lesion", "mole"]
        },
        "Hematologist": {
            "expertise": "blood disorders, anemia, clotting disorders, blood cancers",
            "keywords": ["blood", "anemia", "bleeding", "clotting", "hematology"]
        },
        "Infectious Disease": {
            "expertise": "infections, tropical diseases, immunocompromised conditions",
            "keywords": ["infection", "fever", "tropical", "antibiotic", "virus"]
        },
        "Rheumatologist": {
            "expertise": "autoimmune diseases, arthritis, joint inflammation",
            "keywords": ["arthritis", "autoimmune", "joint pain", "inflammation", "rheumatic"]
        },
        "Oncologist": {
            "expertise": "cancer diagnosis and treatment, chemotherapy, tumor management",
            "keywords": ["cancer", "tumor", "oncology", "chemotherapy", "malignancy"]
        },
        "Emergency Medicine": {
            "expertise": "acute care, trauma, emergency conditions, triage",
            "keywords": ["emergency", "acute", "trauma", "urgent", "critical"]
        },
        "Pediatrician": {
            "expertise": "children's health, developmental issues, pediatric diseases",
            "keywords": ["child", "pediatric", "infant", "developmental", "growth"]
        },
        "Radiologist": {
            "expertise": "medical imaging interpretation, X-rays, CT, MRI, ultrasound",
            "keywords": ["imaging", "X-ray", "CT", "MRI", "radiology", "scan"]
        },
        "Pathologist": {
            "expertise": "disease diagnosis through lab tests, tissue examination, biopsies",
            "keywords": ["biopsy", "lab", "pathology", "tissue", "microscopic"]
        }
    }

    @classmethod
    def get_relevant_specialists(cls, question: str, num_specialists: int = 5) -> List[str]:
        """Get relevant specialists based on question keywords"""
        question_lower = question.lower()
        scores = {}

        for specialty, info in cls.SPECIALTIES.items():
            score = sum(
                1 for keyword in info["keywords"] if keyword in question_lower)
            if score > 0:
                scores[specialty] = score

        # Sort by relevance score
        sorted_specialists = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)
        selected = [spec[0] for spec in sorted_specialists[:num_specialists]]

        # If not enough relevant specialists found, add random ones
        if len(selected) < num_specialists:
            remaining = [s for s in cls.SPECIALTIES.keys()
                         if s not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:num_specialists - len(selected)])

        return selected[:num_specialists]

    @classmethod
    def get_specialist_description(cls, specialty: str) -> str:
        """Get expertise description for a specialist"""
        return cls.SPECIALTIES.get(specialty, {}).get("expertise", "medical expertise")
