# concepts/episode_index.py
"""
Master index of concept inputs for long-form episode generation.
Pass any entry directly to generate_long_form_script().

Usage:
    from concepts.episode_index import CONCEPTS
    concept = CONCEPTS["self_authorship"]
    script = generate_long_form_script(**concept)
"""

CONCEPTS = {

    "self_authorship": {
        "concept_name": "Self-Authorship",
        "concept_description": (
            "The internal capacity to define one's own beliefs, "
            "identity, and social relationships, rather than relying "
            "on external authorities to do so. Developed primarily by "
            "Marcia Baxter Magolda through a longitudinal study of "
            "college students and adults that began in 1986 and "
            "continues today. Built on Robert Kegan's constructive-"
            "developmental theory of orders of consciousness, self-"
            "authorship describes the developmental movement from "
            "following external formulas, through the crossroads of "
            "questioning those formulas, toward authoring one's own "
            "internal foundation. The framework integrates three "
            "developmental dimensions: epistemological (how do I "
            "know?), intrapersonal (who am I?), and interpersonal "
            "(how do I want to construct relationships with others?). "
            "Has become the dominant developmental framework in "
            "higher education and student affairs, shaping learning "
            "partnerships pedagogy, leadership education, and "
            "identity research. Recently challenged by critical "
            "scholars who argue the framework universalizes a "
            "particular cultural pattern of individuation."
        ),
        "earliest_year": 1986,
    },

    "student_departure": {
        "concept_name": "Student Departure and Integration",
        "concept_description": (
            "Vincent Tinto's theoretical model of why students leave "
            "college before completing a degree, first published in "
            "1975 and elaborated in Leaving College (1987, revised "
            "1993). The model argues that student persistence depends "
            "on the degree to which students become integrated into "
            "the academic and social systems of the institution, with "
            "pre-entry attributes, goals, and commitments interacting "
            "with institutional experiences to produce departure "
            "decisions. Drew heavily on Durkheim's theory of suicide "
            "and Van Gennep's rites of passage, applying sociological "
            "frameworks to an educational problem. Became the most "
            "cited theoretical framework in higher education research "
            "and the empirical foundation of institutional retention "
            "practice. Subject to sustained critique from scholars of "
            "color, beginning notably with Tierney's 1992 article and "
            "extending through Museus's CECE model, who argued the "
            "integration concept required minoritized students to "
            "shed cultural identities to belong. The debates over "
            "Tinto's model are themselves a major thread in the "
            "field's intellectual history."
        ),
        "earliest_year": 1975,
    },

    "multiple_dimensions_of_identity": {
        "concept_name": "Multiple Dimensions of Identity",
        "concept_description": (
            "A conceptual model developed by Susan Jones and Marylu "
            "McEwen in 2000 to describe how college students hold "
            "and experience multiple social identities — race, "
            "gender, sexual orientation, class, religion, ability — "
            "simultaneously rather than sequentially. The original "
            "model depicted identity as a core sense of self "
            "surrounded by intersecting identity dimensions whose "
            "salience varies by context. Reconceptualized in 2007 "
            "by Abes, Jones, and McEwen to incorporate Kegan's "
            "meaning-making capacity as a filter through which "
            "contextual influences are processed, accounting for why "
            "students at different developmental positions experience "
            "their identities differently. Represents the field's "
            "primary move beyond single-axis identity development "
            "models toward intersectional analysis. Drew on Kimberlé "
            "Crenshaw's legal scholarship on intersectionality while "
            "translating it into a developmental and psychological "
            "register — a translation that has itself been "
            "critiqued for depoliticizing Crenshaw's original framing."
        ),
        "earliest_year": 2000,
    },

    "campus_racial_climate": {
        "concept_name": "Campus Racial Climate",
        "concept_description": (
            "A multidimensional framework for understanding how the "
            "racial environment of a college campus shapes student "
            "experiences and outcomes, developed primarily by Sylvia "
            "Hurtado and colleagues at UCLA's Higher Education "
            "Research Institute beginning in the early 1990s. The "
            "framework identifies four dimensions of campus climate: "
            "the institution's historical legacy of inclusion or "
            "exclusion, its structural diversity (compositional "
            "representation of different groups), its psychological "
            "climate (perceptions and attitudes), and its behavioral "
            "climate (intergroup interactions). Emerged in the wake "
            "of the 1978 Bakke decision and the affirmative action "
            "debates of the late 1980s and 1990s, providing the "
            "empirical foundation for the diversity-as-educational-"
            "benefit argument that would prove central in Grutter v. "
            "Bollinger (2003). Extended by Samuel Museus into the "
            "Culturally Engaging Campus Environments (CECE) model, "
            "which specifies the institutional conditions that "
            "support success for racially minoritized students. "
            "Remains the primary framework through which diversity "
            "and inclusion work in higher education is theorized "
            "and assessed."
        ),
        "earliest_year": 1992,
    },

    "student_involvement_and_engagement": {
        "concept_name": "Student Involvement and Engagement",
        "concept_description": (
            "Alexander Astin's theory of student involvement, "
            "published in 1984 in the Journal of College Student "
            "Personnel, argued that the amount of physical and "
            "psychological energy a student devotes to the academic "
            "experience is the active ingredient in college impact "
            "— not curriculum content, not institutional "
            "characteristics, but involvement itself. The theory "
            "emerged from Astin's input-environment-output model and "
            "decades of empirical work showing that what students do "
            "in college mattered more than where they went. "
            "Evolved into the broader concept of student engagement, "
            "operationalized through George Kuh's National Survey of "
            "Student Engagement (NSSE) beginning in 1998, which "
            "shifted institutional accountability from inputs and "
            "rankings toward measurable educational practices. "
            "Became the foundation of the high-impact practices "
            "framework and most contemporary student affairs "
            "philosophy. The concept's translation from theory to "
            "measurement to institutional practice is itself a major "
            "story about how higher education research shapes "
            "institutional behavior — and the limits of that influence."
        ),
        "earliest_year": 1984,
    },

    "community_cultural_wealth": {
        "concept_name": "Community Cultural Wealth",
        "concept_description": (
            "Tara Yosso's 2005 reframing of Pierre Bourdieu's "
            "cultural capital theory, published in Race Ethnicity "
            "and Education. Yosso argued that Bourdieu's framework, "
            "as imported into education, had been used to explain "
            "the educational underachievement of communities of "
            "color as a deficit of valued cultural capital. She "
            "inverted the question: instead of asking what "
            "marginalized communities lack, she identified six "
            "forms of capital they bring to educational "
            "settings — aspirational, linguistic, familial, social, "
            "navigational, and resistant capital. The framework "
            "drew on critical race theory's commitment to centering "
            "the experiential knowledge of people of color and "
            "challenged the field's deficit framings of first-"
            "generation, low-income, and minoritized students. "
            "Became one of the most cited pieces in higher "
            "education in the subsequent two decades and the "
            "intellectual foundation for anti-deficit research "
            "frameworks. Its rapid uptake also illustrates how a "
            "compact theoretical move can reshape a field's "
            "vocabulary faster than empirical work alone."
        ),
        "earliest_year": 2005,
    },

    "leadership_identity_development": {
        "concept_name": "Leadership Identity Development",
        "concept_description": (
            "A grounded theory model developed by Susan Komives, "
            "Julie Owen, Susan Longerbeam, Felicia Mainella, and "
            "Laura Osteen, first published in the Journal of College "
            "Student Development in 2005. The model describes how "
            "college students come to see themselves as leaders "
            "through a six-stage developmental process: awareness, "
            "exploration/engagement, leader identified, leadership "
            "differentiated, generativity, and integration/synthesis. "
            "The model's central move was shifting from leadership "
            "as a position or set of behaviors to leadership as an "
            "identity that develops over time through interaction "
            "with peers, mentors, and meaningful involvement. "
            "Emerged from the broader Social Change Model of "
            "Leadership Development (1996) and the post-industrial "
            "leadership paradigm that rejected great-man theories "
            "in favor of relational, collaborative frameworks. Has "
            "become the empirical foundation of leadership "
            "education practice in higher education, shaping how "
            "co-curricular programs, leadership minors, and "
            "fraternity/sorority advising frame their work. The "
            "Multi-Institutional Study of Leadership, beginning in "
            "2006, has provided the largest empirical base for "
            "leadership research in the field's history."
        ),
        "earliest_year": 1996,
    },

    "epistemological_development": {
        "concept_name": "Epistemological Development",
        "concept_description": (
            "The developmental study of how people come to know what "
            "they know, and how that capacity changes through the "
            "college years. Originated with William Perry's 1970 "
            "scheme, developed from interviews with male Harvard "
            "undergraduates, which described nine positions moving "
            "from dualistic right-wrong thinking through multiplistic "
            "acceptance of multiple views to committed relativism. "
            "Critiqued and extended by Mary Belenky, Blythe Clinchy, "
            "Nancy Goldberger, and Jill Tarule in Women's Ways of "
            "Knowing (1986), which argued from interviews with women "
            "that Perry's sequence missed forms of knowing — "
            "particularly silenced and connected knowing — that were "
            "gendered in their distribution and value. Further "
            "developed by Marcia Baxter Magolda's Epistemological "
            "Reflection Model and Patricia King and Karen Strohm "
            "Kitchener's Reflective Judgment Model, both based on "
            "longitudinal studies extending into adulthood. "
            "Represents the cognitive-structural strand of student "
            "development running parallel to psychosocial work, and "
            "the philosophical foundation underneath self-authorship. "
            "Underappreciated by practitioners despite its centrality "
            "to how the field thinks about learning."
        ),
        "earliest_year": 1970,
    },

}


# Suggested generation order for a doctoral student new to the field
RECOMMENDED_ORDER = [
    "student_involvement_and_engagement",  # The widest-reaching concept; orients everything else
    "epistemological_development",          # The cognitive foundation underneath later identity work
    "self_authorship",                      # Where epistemological and psychosocial work integrate
    "multiple_dimensions_of_identity",      # The intersectional turn in identity development
    "student_departure",                    # The retention literature and its critiques
    "campus_racial_climate",                # The institutional environment lens
    "community_cultural_wealth",            # The critical turn most concisely expressed
    "leadership_identity_development",      # Synthesizes developmental + identity + practice
]
