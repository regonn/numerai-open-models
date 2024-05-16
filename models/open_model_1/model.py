import requests
import os
import numerapi
from dotenv import load_dotenv
import pandas as pd
import json
import lightgbm as lgb
import gc
from logging import getLogger
import psutil
import numpy as np


logger = getLogger(__name__)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = os.environ["NUMERAI_MODEL_ID"]


def post_discord(message):
    requests.post(DISCORD_WEBHOOK_URL, {"content": message})


def memory_log_message():
    used = f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB"
    total = f"{psutil.virtual_memory().total / (1024 ** 2):.2f} MB"
    return f"Memory used: {used} / {total}"


def memory_callback(env):
    if env.iteration % 100 == 0 or env.iteration == env.end_iteration - 1:
        logger.info(f"Iteration: {env.iteration}, {memory_log_message()}")


def neutralize(df, target="prediction", by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith("feature")]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1))
    )

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()


post_discord(f"Start: {NUMERAI_MODEL_ID}")
try:
    napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)

    # Download data
    napi.download_dataset("v4.3/train_int8.parquet")
    napi.download_dataset("v4.3/validation_int8.parquet")
    napi.download_dataset("v4.3/live_int8.parquet")
    napi.download_dataset("v4.3/features.json")
    predict_csv_file_name = f"tournament_predictions_{NUMERAI_MODEL_ID}.csv"

    # Load data
    feature_metadata = json.load(open("v4.3/features.json"))
    features = [
        "feature_grave_prevenient_rheotrope",
        "feature_flagging_gadarene_barrymore",
        "feature_heretofore_drowsiest_conjugation",
        "feature_thorniest_laughable_hindustani",
        "feature_bumpier_maidenlike_chordata",
        "feature_canicular_overlong_avocado",
        "feature_third_discreet_solute",
        "feature_skim_expugnable_subception",
        "feature_carolean_tearable_smoothie",
        "feature_intercostal_imbricated_hypothenuse",
        "feature_overeager_pugilistic_diocletian",
        "feature_statesmanlike_tailed_herat",
        "feature_unfertilized_scaldic_partition",
        "feature_galvanizing_whirring_baroscope",
        "feature_veddoid_sport_psychobiology",
        "feature_conveyed_divisional_argemone",
        "feature_undisguised_unenviable_stamen",
        "feature_shrinelike_unreplaceable_nitrogenization",
        "feature_shakier_peskier_transfuser",
        "feature_bantam_matterful_hut",
        "feature_laziest_saronic_hornbeam",
        "feature_eighteen_kafka_segno",
        "feature_battled_premillennial_omelette",
        "feature_dere_isolate_sydney",
        "feature_expended_evitable_darwinian",
        "feature_relinquished_incognizable_batholith",
        "feature_direct_expropriated_harping",
        "feature_coraciiform_sciurine_reef",
        "feature_suspended_intracranial_fischer",
        "feature_unreclaimable_aggregative_meningioma",
        "feature_gamest_zibeline_oakley",
        "feature_observing_vigesimal_completion",
        "feature_crispy_neighboring_jeffersonian",
        "feature_spikiest_anguished_commandership",
        "feature_olden_enchained_leek",
        "feature_roily_frothier_ketene",
        "feature_septuagenary_bully_puna",
        "feature_disintegrable_snakier_zion",
        "feature_cloaked_taillike_usurpation",
        "feature_scalding_assumptive_sentimentalist",
        "feature_incredible_sipunculid_midriff",
        "feature_acclimatisable_unfeigned_maghreb",
        "feature_histie_undeified_applicability",
        "feature_unscriptural_coconut_trisulphide",
        "feature_unqualifying_pursuant_antihistamine",
        "feature_porkiest_waspy_recycling",
        "feature_cragged_sacred_malabo",
        "feature_deposed_toughish_bribery",
        "feature_mesmerized_springing_euchologion",
        "feature_indicial_caryatidal_kendal",
        "feature_confabulatory_malarian_phenotype",
        "feature_antiperistaltic_tribadic_brigand",
        "feature_opposing_intracardiac_delimitation",
        "feature_reposeful_apatetic_trudeau",
        "feature_ironfisted_nonvintage_chlorpromazine",
        "feature_interconnected_greige_mohammedan",
        "feature_unhorsed_morphogenetic_affusion",
        "feature_virescent_telugu_neighbour",
        "feature_zoophoric_underglaze_algin",
        "feature_jeffersonian_subjacent_petrification",
        "feature_variolate_reducible_sweet",
        "feature_demountable_unprejudiced_neighbourhood",
        "feature_asinine_unsatiable_avion",
        "feature_meningeal_realizing_smoke",
        "feature_vaccinated_compellable_schizont",
        "feature_consuetudinary_relivable_monad",
        "feature_piggie_pearliest_couvade",
        "feature_surefooted_quakier_constructivism",
        "feature_thymic_formidable_misericord",
        "feature_godliest_consistorian_woodpecker",
        "feature_frisky_transuranic_spire",
        "feature_unsapped_anionic_catherine",
        "feature_financed_sulphuretted_libertinage",
        "feature_fungible_allotted_deterioration",
        "feature_azimuthal_disconcerted_bock",
        "feature_lackluster_thermic_bovid",
        "feature_exhibitionist_bicuspidate_goalpost",
        "feature_floodlighted_apprentice_comstockery",
        "feature_unpitied_jingoist_pyretology",
        "feature_virtuoso_gasping_studwork",
        "feature_thermophile_noisette_swamper",
        "feature_muggiest_explicit_barnardo",
        "feature_endoplasmic_inwrought_percival",
        "feature_doggone_seeable_mask",
        "feature_pulmonic_bladed_affray",
        "feature_unreaving_intensive_docudrama",
        "feature_suspensory_unrecounted_transcendent",
        "feature_unweary_avionic_claudine",
        "feature_alert_eddic_semicylinder",
        "feature_navigational_enured_condensability",
        "feature_proxy_transcriptive_scaler",
        "feature_lovelorn_aided_limiter",
        "feature_depilatory_tin_trinity",
        "feature_anglophobic_unformed_maneuverer",
        "feature_detractive_samoa_expiration",
        "feature_diversified_adventive_peridinium",
        "feature_hirsute_corkier_beldame",
        "feature_chunky_fallen_erasure",
        "feature_geostrophic_adaptative_karla",
        "feature_undescended_crawly_armet",
        "feature_psychrometrical_inexpensive_opah",
        "feature_productile_auriform_fil",
        "feature_subatomic_raffish_hexagram",
        "feature_spongy_antarctic_passel",
        "feature_weightiest_protozoic_brawler",
        "feature_distal_indented_modernity",
        "feature_waxing_jaggy_bondswoman",
        "feature_professional_platonic_marten",
        "feature_verticillate_deflective_toweling",
        "feature_unwitty_fauve_altimeter",
        "feature_renascent_scopate_marrowfat",
        "feature_heterotrophic_anechoic_annexationist",
        "feature_china_fistular_phenylketonuria",
        "feature_comparative_sottish_delivery",
        "feature_heartfelt_laddish_cuyp",
        "feature_katabolic_peridotic_ergotism",
        "feature_prefigurative_downstream_transvaluation",
        "feature_declinatory_unfilmed_lutheranism",
        "feature_brief_optimistic_consentaneity",
        "feature_wealthy_asinine_beduin",
        "feature_ministrative_unvocalised_truffle",
        "feature_jazziest_spellbinding_philabeg",
        "feature_healthier_unconnected_clave",
        "feature_pronominal_billowing_semeiology",
        "feature_unwilled_exhortative_gisarme",
        "feature_crackly_ripuarian_parure",
        "feature_undrilled_wheezier_countermand",
        "feature_concentric_gubernatorial_grandeur",
        "feature_undepreciated_partitive_ipomoea",
        "feature_mammonistic_smeared_stigma",
        "feature_dure_jaspery_mugging",
        "feature_translative_quantitative_eschewer",
        "feature_wistful_tussive_cycloserine",
        "feature_unpassable_wedgy_blossom",
        "feature_botchier_universalistic_nullifier",
        "feature_correspondent_orderly_personalisation",
        "feature_holy_chic_cali",
        "feature_bronchitic_miscible_inwall",
        "feature_incredible_plane_sacque",
        "feature_biological_caprine_cannoneer",
        "feature_gubernacular_liguloid_frankie",
        "feature_procedural_approximal_centimeter",
        "feature_wannish_record_lunette",
        "feature_expedited_yucky_anesthetic",
        "feature_unsensing_enterprising_janissary",
        "feature_lepidote_malevolent_maori",
        "feature_applicative_cometic_gimp",
        "feature_marxist_unstoppable_eutherian",
        "feature_zygodactyl_exponible_lathi",
        "feature_thinkable_pledged_marten",
        "feature_unpoetic_stretchy_pakeha",
        "feature_uncurtailed_sabaean_ode",
        "feature_hippiatric_tinctorial_slowpoke",
        "feature_homespun_brinish_woof",
        "feature_phylogenetic_paramount_caperer",
        "feature_chauvinistic_irksome_colloquy",
        "feature_polypoid_taken_upgrader",
        "feature_disaffected_formulism_tabbouleh",
        "feature_decompressive_genotypic_homoptera",
        "feature_uneasy_unaccommodating_immortality",
        "feature_apophthegmatic_phenotypic_bother",
        "feature_damask_tabu_cobweb",
        "feature_styloid_subdermal_cytotoxin",
        "feature_affettuoso_taxidermic_greg",
        "feature_taxable_homogenized_rhodian",
        "feature_intimidatory_wight_resource",
        "feature_intoed_pedigree_canful",
        "feature_winking_phlegmier_intro",
        "feature_urnfield_linty_strawman",
        "feature_unpressed_mahratta_dah",
        "feature_expectative_intimidated_bluffer",
        "feature_lucullian_unshunned_ulex",
        "feature_chastised_antitypical_palooka",
        "feature_saclike_hyphal_postulator",
        "feature_surpliced_unachievable_nubecula",
        "feature_dissipated_dressiest_bombast",
        "feature_sesquicentennial_rubbliest_kremlinologist",
        "feature_phrenitic_foldable_trussing",
        "feature_distressed_bloated_disquietude",
        "feature_airy_divaricate_allomorph",
        "feature_jansenism_perceptual_quandong",
        "feature_ugrian_schizocarpic_skulk",
        "feature_closing_branchy_kirman",
        "feature_geophytic_penitential_deutzia",
        "feature_miasmal_mozartian_gervase",
        "feature_numeral_cagey_haulm",
        "feature_gravimetric_ski_enigma",
        "feature_aneroid_cufic_prolonge",
        "feature_unsentenced_disapproved_pastorship",
        "feature_libidinal_guardable_siderite",
        "feature_ruthenic_peremptory_truth",
        "feature_subminiature_catchable_classic",
        "feature_recent_shorty_preferment",
        "feature_recreational_homiletic_nubian",
        "feature_participating_unrecollected_braiding",
        "feature_fatalistic_cramoisy_locative",
        "feature_conic_spread_semisolid",
        "feature_unsurveyed_hymenal_cheapskate",
        "feature_associate_unproper_gridder",
        "feature_reservable_peristomal_emden",
        "feature_undersexed_hedonic_spew",
        "feature_epicontinental_centum_raine",
        "feature_leaky_overloaded_rhodium",
        "feature_martial_hallowed_incorruptibility",
        "feature_vixenish_nodose_phocomelia",
        "feature_unperceivable_unrumpled_appendant",
        "feature_imperishable_mightiest_emulsoid",
        "feature_earthier_adjacent_hydropathy",
        "feature_recommendatory_prissy_flutter",
        "feature_anthracoid_unforeseen_arizonan",
        "feature_antarthritic_syzygial_wonderland",
        "feature_demonic_eocene_polygamy",
        "feature_enfeebled_undiplomatic_stereoscope",
        "feature_star_roomier_mapping",
        "feature_additive_untrustworthy_hierologist",
        "feature_astonied_substitutionary_mage",
        "feature_gandhian_discretional_cricoid",
        "feature_eclamptic_unblissful_dip",
        "feature_minuscule_confusing_flaunter",
        "feature_devoted_sollar_whispering",
        "feature_revitalizing_rutilant_swastika",
        "feature_distractible_unreposing_arrowhead",
        "feature_platy_idiotic_vladimir",
        "feature_ataractic_swept_rubeola",
        "feature_garlicky_allopatric_sarcocarp",
    ]
    logger.info(f"Using {len(features)} features")
    logger.info(memory_log_message())

    # Train data
    train = pd.read_parquet(
        "v4.3/train_int8.parquet", columns=["era"] + features + ["target"]
    )
    logger.info(f"Loaded {len(train)} rows of training data")
    logger.info(memory_log_message())

    dtrain = lgb.Dataset(train[features], label=train["target"])
    del train
    gc.collect()
    logger.info("Create LGBM Dataset")
    logger.info(memory_log_message())

    # Validation data
    validation = pd.read_parquet(
        "v4.3/validation_int8.parquet",
        columns=["era", "data_type"] + features + ["target"],
    )
    validation = validation[validation["data_type"] == "validation"]
    logger.info(f"Loaded {len(validation)} rows of validation data")
    logger.info(memory_log_message())

    dvalid = lgb.Dataset(validation[features], label=validation["target"])
    del validation
    gc.collect()
    logger.info("Create LGBM Dataset")
    logger.info(memory_log_message())

    # Train model
    logger.info("Start training")
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 2**6 - 1,
        "colsample_bytree": 0.1,
        "random_state": 46,  # sexy random seed
        "force_col_wise": True,
    }
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            memory_callback,
        ],
    )
    logger.info("end training")
    logger.info(memory_log_message())

    # Predict
    live_features = pd.read_parquet("v4.3/live_int8.parquet", columns=features)
    live_predictions = model.predict(
        live_features[features], num_iteration=model.best_iteration
    )
    live_features["prediction"] = live_predictions

    submission = neutralize(
        live_features, target="prediction", by=None, proportion=1.0
    ).rank(pct=True)

    # Submit
    submission.to_frame("prediction").to_csv(predict_csv_file_name, index=True)
    model_id = napi.get_models()[NUMERAI_MODEL_ID]
    napi.upload_predictions(predict_csv_file_name, model_id=model_id)

    post_discord(f"Submit Success: {NUMERAI_MODEL_ID}")
except Exception as e:
    post_discord(f"Numerai Submit Failure. Model: {NUMERAI_MODEL_ID}, Error: {str(e)}")
