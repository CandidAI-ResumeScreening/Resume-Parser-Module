import pandas as pd
import re
from typing import List, Dict

def load_skills(skills_csv_path: str) -> List[str]:
    """
    Load and enhance skills from a CSV file with additional diverse skills.
    """
    skills_df = pd.read_csv(skills_csv_path)
    raw_skills = [skill.strip().lower() for skill in skills_df.columns]

    diverse_skills = [
        # Soft Skills
        "teamwork", "communication", "leadership", "adaptability", "creativity", "hardwork",
        "critical thinking", "time management", "collaboration", "problem solving", "attention to detail",

        # Office Tools
        "excel", "powerpoint", "word", "outlook", "notion", "trello", "slack", "asana", "miro", "google docs", "confluence", "figma", "canva",

        # Design & Media
        "photoshop", "illustrator", "after effects", "premiere pro", "lightroom",
        "davinci resolve", "blender", "audacity", "capcut",

        # Data & BI
        "tableau", "power bi", "qlikview", "alteryx", "pentaho", "knime", "apache nifi", "dbt",

        # Marketing
        "google analytics", "semrush", "ahrefs", "mailchimp", "hubspot", "buffer", "hootsuite", "wordpress",

        # Engineering
        "autocad", "revit", "solidworks", "matlab", "simulink", "arduino", "raspberry pi",

        # Education
        "research", "tutoring", "lesson planning", "classroom management", "curriculum development",

        # HR
        "recruitment", "performance reviews", "payroll", "employee engagement", "hrms"
    ]
    # Categorized skills
    categorized_skills: Dict[str, List[str]] = {
        'Programming Language': [
            'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'go',
            'scala', 'r', 'matlab', 'typescript', 'rust', 'perl', 'bash', 'powershell', 'sql', 'html', 'css', 'dart', 'groovy', 'lua'
        ],
        'Framework/Library': [
            'django', 'flask', 'fastapi', 'spring', 'react', 'angular', 'vue', 'node.js', 'express',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'bootstrap', 'jquery',
            'laravel', 'rails', 'asp.net', 'flutter', 'xamarin', 'react native'
        ],
        'Cloud/DevOps': [
            'hadoop', 'spark', 'kubernetes', 'docker', 'aws', 'azure', 'gcp', 'firebase',
            'selenium', 'jenkins', 'terraform', 'ansible', 'chef', 'puppet', 'circleci',
            'travis ci', 'github actions', 'amazon web services', 'google cloud'
        ],
        'Database': [
            'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'sql server', 'redis',
            'cassandra', 'dynamodb', 'mariadb', 'elasticsearch', 'neo4j'
        ],
        'Tool/Software': [
            'figma', 'sketch', 'photoshop', 'illustrator', 'indesign', 'xd', 'jira', 'trello',
            'asana', 'git', 'github', 'gitlab', 'bitbucket', 'confluence', 'notion', 'slack',
            'tableau', 'power bi', 'excel'
        ],
        'Methodology': [
            'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'ci/cd',
            'test driven development', 'tdd', 'behavior driven development', 'bdd'
        ],
        'AI/ML': [
            'machine learning', 'deep learning', 'natural language processing', 'nlp',
            'computer vision', 'data science', 'neural networks', 'ai',
            'artificial intelligence', 'reinforcement learning', 'data mining'
        ],
        'Soft Skill': [
            'communication', 'teamwork', 'problem solving', 'critical thinking', 'leadership',
            'time management', 'adaptability', 'creativity', 'emotional intelligence',
            'conflict resolution', 'presentation', 'negotiation', 'decision making',
            'project management', 'mentoring', 'coaching', 'analytical skills',
            'attention to detail', 'organization', 'flexibility', 'interpersonal skills',
            'collaboration', 'innovation', 'work ethic', 'customer service',
            'active listening', 'research'
        ]
    }

    # Flatten categorized skills to list
    added_skills = [skill.lower() for skills in categorized_skills.values() for skill in skills]

    # Combine all and deduplicate
    enhanced_skills = list(set(raw_skills + diverse_skills + added_skills))

    return enhanced_skills

def extract_skills_from_text(text: str, skills_list: List[str]) -> List[str]:
    """
    Extract skills from resume text using simple pattern matching.
    """
    text = text.lower()
    tokens = re.findall(r'\b[\w\-\+\.#]+\b', text)

    # Generate phrases for matching multi-word skills
    phrases = tokens + [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)] + [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]

    matched_skills = list(set(skill for skill in skills_list if skill in phrases))

    return matched_skills
